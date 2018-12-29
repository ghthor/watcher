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
	"time"

	"github.com/fluxio/multierror"
	"github.com/hybridgroup/mjpeg"
	"gocv.io/x/gocv"
)

const ESC_KEY = 27

const DefaultDevice = "/dev/video0"
const DefaultWidth = 960
const DefaultHeight = 720

// TODO: Does this need to be a ratio based on image WxH?
const MinimumArea = 3000

const DefaultWatchTick = 200 * time.Millisecond

const DefaultHost = "localhost:8088"

var (
	ColorRed   = color.RGBA{255, 0, 0, 0}
	ColorGreen = color.RGBA{0, 255, 0, 0}
	ColorBlue  = color.RGBA{0, 0, 255, 0}
)

// Model Program States
type State interface {
	Execute() *State
}

type Frame struct {
	frame,
	frameDelta,
	frameThresh,
	frameDebug gocv.Mat

	mog2 gocv.BackgroundSubtractorMOG2
}

func NewFrame() *Frame {
	return &Frame{
		frame:       gocv.NewMat(),
		frameDelta:  gocv.NewMat(),
		frameThresh: gocv.NewMat(),
		frameDebug:  gocv.NewMat(),

		mog2: gocv.NewBackgroundSubtractorMOG2(),
	}
}

func (img *Frame) Close() error {
	var e multierror.Accumulator
	e.Push(img.frame.Close())
	e.Push(img.frameDelta.Close())
	e.Push(img.frameThresh.Close())
	e.Push(img.frameDebug.Close())
	e.Push(img.mog2.Close())
	return e.Error()
}

func (img *Frame) FindContours() [][]image.Point {
	// Cleaning up image
	// Phase 1: obtain foreground only
	img.mog2.Apply(img.frame, &img.frameDelta)

	// Phase 2: use threshold
	gocv.Threshold(img.frameDelta, &img.frameThresh, 25, 255, gocv.ThresholdBinary)

	// Phase 3: dilate
	kernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
	defer kernel.Close()
	gocv.Dilate(img.frameThresh, &img.frameThresh, kernel)

	return gocv.FindContours(img.frameThresh, gocv.RetrievalExternal, gocv.ChainApproxSimple)
}

type Watching struct {
	*Frame
}

type MotionDetected struct {
	*Frame
}

type Recording struct {
	*Frame
}

func main() {
	devicePath := DefaultDevice

	webcam, err := gocv.OpenVideoCapture(devicePath)
	if err != nil {
		fmt.Printf("Error opening video capture device: %v\n", devicePath)
		return
	}
	defer webcam.Close()

	webcam.Set(gocv.VideoCaptureFrameWidth, DefaultWidth)
	webcam.Set(gocv.VideoCaptureFrameHeight, DefaultHeight)
	log.Printf("opened %vx%v", webcam.Get(gocv.VideoCaptureFrameHeight), webcam.Get(gocv.VideoCaptureFrameWidth))

	window := gocv.NewWindow("Watcher")
	defer window.Close()

	img := NewFrame()
	defer img.Close()

	// create the mjpeg debugStream
	debugStream := mjpeg.NewStream()

	go func() {
		fmt.Println("Debug stream to http://" + DefaultHost)
		// start http server
		http.Handle("/", debugStream)
		log.Fatal(http.ListenAndServe(DefaultHost, nil))
	}()

	status := "Watching"
	statusColor := ColorGreen
	watchTick := time.Tick(200 * time.Millisecond)

	fmt.Printf("Start reading camera device: %v\n", devicePath)
	for {
		select {
		case <-watchTick:
			goto readImg
		}

	readImg:
		if ok := webcam.Read(&img.frame); !ok {
			fmt.Printf("Error cannot read device %d\n", devicePath)
			return
		}
		if img.frame.Empty() {
			goto readImg
		}

		img.frame.CopyTo(&img.frameDebug)

		status = "Watching"
		statusColor = ColorGreen

		contours := img.FindContours()
		for i, c := range contours {
			area := gocv.ContourArea(c)
			if area < MinimumArea {
				continue
			}

			status = "Recording"
			statusColor = ColorRed
			gocv.DrawContours(&img.frameDebug, contours, i, statusColor, 2)

			rect := gocv.BoundingRect(c)
			gocv.Rectangle(&img.frameDebug, rect, ColorBlue, 2)
		}

		gocv.PutText(&img.frameDebug, status, image.Pt(10, 20), gocv.FontHersheyPlain, 1.2, statusColor, 2)

		// TODO: Move to a context
		buf, _ := gocv.IMEncode(".jpg", img.frameDebug)
		debugStream.UpdateJPEG(buf)

		switch status {
		case "Recording":
			goto readImg
		case "Watching":
			continue
		default:
			continue
		}
	}
}
