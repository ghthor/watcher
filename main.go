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

const ESC_KEY = 27

const DefaultDevice = 0
const DefaultWidth = 960
const DefaultHeight = 720

// TODO: Does this need to be a ratio based on image WxH?
const MinimumArea = 3000

const DefaultWatchCycle = 200 * time.Millisecond
const DefaultRecordingThreshold = 3 * time.Second

const DefaultHost = "localhost:8088"

var (
	ColorRed   = color.RGBA{255, 0, 0, 0}
	ColorGreen = color.RGBA{0, 255, 0, 0}
	ColorBlue  = color.RGBA{0, 0, 255, 0}
)

var ErrReadDevice = errors.New("error reading device")

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
	// 	<-time.After(20 * time.Second)
	// 	log.Fatal("demo timeout")
	// }()

	go func() {
		fmt.Println("Debug stream to http://" + DefaultHost)
		// start http server
		http.Handle("/", debugStream)
		log.Fatal(http.ListenAndServe(DefaultHost, nil))
	}()

	frame := gocv.NewMat()
	frameDelta := gocv.NewMat()
	frameThresh := gocv.NewMat()
	frameDebug := gocv.NewMat()
	mog2 := gocv.NewBackgroundSubtractorMOG2()
	defer func() {
		var e multierror.Accumulator
		e.Push(frame.Close())
		e.Push(frameDelta.Close())
		e.Push(frameThresh.Close())
		e.Push(frameDebug.Close())
		e.Push(mog2.Close())
		if e.Error() != nil {
			log.Fatal(e.Error())
		}
	}()

	motionDetected := false
	motionDetectedAt := time.Now()
	motionFrameCount := 0
	recording := false

	watchDelay := time.Tick(DefaultWatchCycle)

	fmt.Printf("Start reading camera device: %v\n", devicePath)
	for {
		motionDetected = false
		if ok := webcam.Read(&frame); !ok {
			fmt.Printf("Error cannot read device %d\n", devicePath)
		}
		if frame.Empty() {
			continue
		}

		frame.CopyTo(&frameDebug)
		// Cleaning up image
		// Phase 1: obtain foreground only
		mog2.Apply(frame, &frameDelta)
		// Phase 2: use threshold
		gocv.Threshold(frameDelta, &frameThresh, 25, 255, gocv.ThresholdBinary)
		// Phase 3: dilate
		kernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
		gocv.Dilate(frameThresh, &frameThresh, kernel)

		shapes := gocv.FindContours(frameThresh, gocv.RetrievalExternal, gocv.ChainApproxSimple)
		for i, shape := range shapes {
			area := gocv.ContourArea(shape)
			if area < MinimumArea {
				continue
			}

			motionDetected = true
			gocv.DrawContours(&frameDebug, shapes, i, ColorRed, 2)

			rect := gocv.BoundingRect(shape)
			gocv.Rectangle(&frameDebug, rect, ColorBlue, 2)
		}

		kernel.Close()

		switch {
		case !motionDetected:
			motionDetected = false
			motionFrameCount = 0
			recording = false
		case motionDetected && motionFrameCount == 0:
			motionDetectedAt = time.Now()
			fallthrough
		default:
			motionFrameCount++
		}

		recording = motionDetected && time.Now().After(motionDetectedAt.Add(DefaultRecordingThreshold))

		if !motionDetected {
			gocv.PutText(
				&frameDebug,
				fmt.Sprint("Watching"),
				image.Pt(10, 20),
				gocv.FontHersheyPlain, 1.2, ColorGreen, 2)
		} else if motionDetected && !recording {
			gocv.PutText(
				&frameDebug,
				fmt.Sprint("Motion Detected:", motionFrameCount),
				image.Pt(10, 20),
				gocv.FontHersheyPlain, 1.2, ColorRed, 2)
		} else {
			gocv.PutText(
				&frameDebug,
				fmt.Sprint("Recording:", motionFrameCount),
				image.Pt(10, 20),
				gocv.FontHersheyPlain, 1.2, ColorRed, 2)
		}

		buf, _ := gocv.IMEncode(".jpg", frameDebug)
		debugStream.UpdateJPEG(buf)

		if !motionDetected {
			select {
			case <-watchDelay:
			}
		}
	}

}
