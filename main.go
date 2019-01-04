// What it does:
//
// This example detects motion using a delta threshold from the first frame,
// and then finds contours to determine where the object is located.
//
// How to run:
//
// 		go run main.go
//

package main

import (
	"context"
	"errors"
	"fmt"
	"image"
	"image/color"
	"log"
	"net/http"
	"os/exec"
	"time"

	"github.com/fluxio/multierror"
	"github.com/hybridgroup/mjpeg"
	"gocv.io/x/gocv"
)

const FilenameLayout = "2006-01-02-150405-MST"

const ESC_KEY = 27

const DefaultDevice = 0
const DefaultWidth = 960
const DefaultHeight = 720

// TODO: Does this need to be a ratio based on image WxH?
const MinimumArea = 3000

const DefaultWatchCycle = 200 * time.Millisecond

// TODO: Fix Names
const DefaultRecordingThreshold = 2 * time.Second
const DefaultRecordingDropoff = 2 * time.Second

const DefaultHost = "localhost:8088"

var (
	ColorRed   = color.RGBA{255, 0, 0, 0}
	ColorGreen = color.RGBA{0, 255, 0, 0}
	ColorBlue  = color.RGBA{0, 0, 255, 0}
)

var ErrReadDevice = errors.New("error reading device")

type Watcher struct {
	src *gocv.VideoCapture
	dst *mjpeg.Stream

	img,
	imgDelta,
	imgThresh,
	imgDebug gocv.Mat

	mog2 gocv.BackgroundSubtractorMOG2

	motionDetectedAt time.Time
}

func NewWatcher(src *gocv.VideoCapture, dst *mjpeg.Stream) *Watcher {
	return &Watcher{
		src: src,
		dst: dst,

		img:       gocv.NewMat(),
		imgDelta:  gocv.NewMat(),
		imgThresh: gocv.NewMat(),
		imgDebug:  gocv.NewMat(),

		mog2: gocv.NewBackgroundSubtractorMOG2(),
	}
}

func (w *Watcher) Close() error {
	var e multierror.Accumulator
	e.Push(w.img.Close())
	e.Push(w.imgDelta.Close())
	e.Push(w.imgThresh.Close())
	e.Push(w.imgDebug.Close())
	e.Push(w.mog2.Close())
	return e.Error()
}

func (w *Watcher) Read(ctx context.Context) error {
	for {
		if ok := w.src.Read(&w.img); !ok {
			return ErrReadDevice
		}
		if w.img.Empty() {
			continue
		}

		w.img.CopyTo(&w.imgDebug)
		return nil
	}
}

func (w *Watcher) FindContours() [][]image.Point {
	// TODO: [+] Quadtree Debug Image with each phase
	// Cleaning up image
	// Phase 1: obtain foreground only
	w.mog2.Apply(w.img, &w.imgDelta)

	// Phase 2: use threshold
	gocv.Threshold(w.imgDelta, &w.imgThresh, 25, 255, gocv.ThresholdBinary)

	// Phase 3: dilate
	kernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
	defer kernel.Close()
	gocv.Dilate(w.imgThresh, &w.imgThresh, kernel)

	return gocv.FindContours(w.imgThresh, gocv.RetrievalExternal, gocv.ChainApproxSimple)
}

func (w *Watcher) FindMotion() bool {
	motionDetected := false
	shapes := w.FindContours()
	for i, shape := range shapes {
		area := gocv.ContourArea(shape)
		if area < MinimumArea {
			continue
		}

		motionDetected = true

		rect := gocv.BoundingRect(shape)
		gocv.Rectangle(&w.imgDebug, rect, ColorBlue, 2)
		gocv.DrawContours(&w.imgDebug, shapes, i, ColorRed, 2)
	}

	if motionDetected {
		w.motionDetectedAt = time.Now()
	}

	return motionDetected
}

func (w *Watcher) PutText(img *gocv.Mat, unixDate string, msg string, msgColor color.RGBA) {
	//TODO: Calculate FPS
	gocv.PutText(img, unixDate,
		image.Pt(10, 20), gocv.FontHersheyPlain, 1.2, msgColor, 2)
	gocv.PutText(img, msg,
		image.Pt(10, 50), gocv.FontHersheyPlain, 1.2, msgColor, 2)
}

func (w *Watcher) UpdateDebugStream() {
	imgDebug, _ := gocv.IMEncode(".jpg", w.imgDebug)
	w.dst.UpdateJPEG(imgDebug)
}

type WatcherKernel func(context.Context) (WatcherKernel, error)

func (w *Watcher) Watching(ctx context.Context) (WatcherKernel, error) {
	readLimiter := time.NewTicker(DefaultWatchCycle)
	defer readLimiter.Stop()

	for {
		err := w.Read(ctx)
		if err != nil {
			return nil, err
		}
		motionDetected := w.FindMotion()

		switch {
		case motionDetected:
			return w.MotionDetected, nil
		default:
		}

		w.PutText(&w.imgDebug, time.Now().Format(time.UnixDate), "Watching", ColorGreen)
		w.UpdateDebugStream()
		<-readLimiter.C
	}
}

func (w *Watcher) BackToWatching(ctx context.Context) (WatcherKernel, error) {
	w.PutText(&w.imgDebug, time.Now().Format(time.UnixDate), "Watching", ColorGreen)
	w.UpdateDebugStream()
	return w.Watching, nil

}

func (w *Watcher) MotionDetected(ctx context.Context) (WatcherKernel, error) {
	motionBegan := time.Now()
	for {
		w.PutText(&w.imgDebug, time.Now().Format(time.UnixDate), "Motion Detected", ColorGreen)
		w.UpdateDebugStream()

		err := w.Read(ctx)
		if err != nil {
			return nil, err
		}
		motionDetected := w.FindMotion()

		switch {
		case !motionDetected:
			return w.BackToWatching, nil

		case motionDetected && time.Since(motionBegan) > DefaultRecordingThreshold:
			return w.Recording, nil

		default:
			continue
		}
	}
}

func (w *Watcher) Recording(ctx context.Context) (WatcherKernel, error) {
	recordingBegan := time.Now()
	log.Print("Recording Began: ", recordingBegan.Format(time.UnixDate))

	filename := fmt.Sprint(recordingBegan.Format(FilenameLayout), ".mp4")
	file, err := gocv.VideoWriterFile(filename, "avc1", 25, w.img.Cols(), w.img.Rows(), true)
	if err != nil {
		return w.BackToWatching, err
	}
	defer func() {
		go func() {
			if err := file.Close(); err != nil {
				log.Print(err)
			}
		}()
	}()

	now := time.Now().Format(time.UnixDate)
	w.PutText(&w.img,
		now, "Recording", ColorRed)
	w.PutText(&w.imgDebug,
		now, "Recording", ColorRed)

	err = file.Write(w.img)
	if err != nil {
		return w.BackToWatching, err
	}
	w.UpdateDebugStream()

	for {
		err := w.Read(ctx)
		if err != nil {
			return nil, err
		}
		motionDetected := w.FindMotion()

		if !motionDetected && time.Since(w.motionDetectedAt) > DefaultRecordingDropoff {
			return w.BackToWatching, nil
		}

		now := time.Now()
		nowStr := fmt.Sprintf("%s %s", now.Format(time.UnixDate), now.Sub(recordingBegan).Round(time.Second))
		if !motionDetected {
			w.PutText(&w.img, nowStr, "Recording", ColorGreen)
			w.PutText(&w.imgDebug, nowStr, "Recording", ColorGreen)
		} else {
			w.PutText(&w.img, nowStr, "Recording", ColorRed)
			w.PutText(&w.imgDebug, nowStr, "Recording", ColorRed)
		}

		err = file.Write(w.img)
		if err != nil {
			return w.BackToWatching, err
		}
		w.UpdateDebugStream()
	}
}

func main() {
	devicePath := DefaultDevice

	ErrReadDevice = errors.New(fmt.Sprint("error reading device ", devicePath))
	input, err := gocv.OpenVideoCapture(devicePath)
	if err != nil {
		log.Fatalf("Error opening video capture device: %v\n", devicePath)
		return
	}
	defer input.Close()

	input.Set(gocv.VideoCaptureFrameWidth, DefaultWidth)
	input.Set(gocv.VideoCaptureFrameHeight, DefaultHeight)
	log.Printf("opened %vx%v", input.Get(gocv.VideoCaptureFrameHeight), input.Get(gocv.VideoCaptureFrameWidth))
	debugStream := mjpeg.NewStream()

	go func() {
		<-time.After(500 * time.Millisecond)
		cmd := exec.Command("open", "http://"+DefaultHost)
		if err := cmd.Run(); err != nil {
			log.Fatal(err)
		}
	}()

	// go func() {
	// 	<-time.After(5 * time.Second)
	// 	log.Fatal("debug timeout")
	// }()

	go func() {
		fmt.Println("Debug stream to http://" + DefaultHost)
		// start http server
		http.Handle("/", debugStream)
		log.Fatal(http.ListenAndServe(DefaultHost, nil))
	}()

	watcher := NewWatcher(input, debugStream)
	defer func() {
		if err := watcher.Close(); err != nil {
			log.Fatal(err)
		}
	}()

	kernel := watcher.Watching

	for kernel != nil {
		kernel, err = kernel(context.Background())
		if err != nil {
			log.Fatal(err)
		}
	}
}
