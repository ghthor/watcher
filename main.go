// What it does:
//
// This example detects motion using a delta threshold from the first frame,
// and then finds contours to determine where the object is located.
//
// How to run:
//
// 		go run ./demo/motion-detect/motion/main.go 0
//

package main

import (
	"fmt"
	"image"
	"image/color"
	"log"
	"time"

	"gocv.io/x/gocv"
)

const ESC_KEY = 27

const DefaultDevice = "/dev/video0"
const DefaultWidth = 960
const DefaultHeight = 720

// TODO: Does this need to be a ratio based on image WxH?
const MinimumArea = 3000

const DefaultWatchTick = 200 * time.Millisecond

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

	img := gocv.NewMat()
	defer img.Close()

	imgDelta := gocv.NewMat()
	defer imgDelta.Close()

	imgThresh := gocv.NewMat()
	defer imgThresh.Close()

	mog2 := gocv.NewBackgroundSubtractorMOG2()
	defer mog2.Close()

	status := "Watching"
	statusColor := color.RGBA{0, 255, 0, 0}
	watchTick := time.Tick(200 * time.Millisecond)

	fmt.Printf("Start reading camera device: %v\n", devicePath)
	for {
		select {
		case <-watchTick:
			goto readImg
		}

	readImg:
		if ok := webcam.Read(&img); !ok {
			fmt.Printf("Error cannot read device %d\n", devicePath)
			return
		}
		if img.Empty() {
			goto readImg
		}

		status = "Watching"

		// first phase of cleaning up image, obtain foreground only
		mog2.Apply(img, &imgDelta)

		// remaining cleanup of the image to use for finding contours.
		// first use threshold
		gocv.Threshold(imgDelta, &imgThresh, 25, 255, gocv.ThresholdBinary)

		// then dilate
		kernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
		defer kernel.Close()
		gocv.Dilate(imgThresh, &imgThresh, kernel)

		// now find contours
		contours := gocv.FindContours(imgThresh, gocv.RetrievalExternal, gocv.ChainApproxSimple)
		for i, c := range contours {
			area := gocv.ContourArea(c)
			if area < MinimumArea {
				continue
			}

			status = "Recording"
			gocv.DrawContours(&img, contours, i, statusColor, 2)

			rect := gocv.BoundingRect(c)
			gocv.Rectangle(&img, rect, color.RGBA{0, 0, 255, 0}, 2)
		}

		switch status {
		case "Recording":
			statusColor = color.RGBA{255, 0, 0, 0}
		default:
			statusColor = color.RGBA{255, 0, 0, 0}
		}
		gocv.PutText(&img, status, image.Pt(10, 20), gocv.FontHersheyPlain, 1.2, statusColor, 2)

		// TODO: Move to a context
		window.IMShow(img)
		if window.WaitKey(1) == ESC_KEY {
			break
		}

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
