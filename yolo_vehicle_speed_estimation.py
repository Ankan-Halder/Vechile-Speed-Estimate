from support.centroidtracker import CentroidTracker
from support.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import math
import os

vechiles =['car','motorbike','bus','truck','bicycle','train']


#input output and yolo directory
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=False,
	help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
ap.add_argument("-s", "--skip-frames", type=int, default=10,
	help="# of skip frames between detections")

args = vars(ap.parse_args())


# estimation of speed using mathematical formula

def estimateSpeed(location1, location2, ppm, fs):
	d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
	d_meters = d_pixels/ppm
	speed = d_meters*fs*3.6
	return speed

# loading of yolov3 
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# input the video 
if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])
fs = vs.get(cv2.CAP_PROP_FPS)

writer = None
(W, H) = (None, None)


# counting of the frames
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# calling the centriod tracker function

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableOjects = {}

totalFrames = 0
fps = FPS().start()

while True:
    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    frame = imutils.resize(frame, width=1024)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# taking the height and width of the image 
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

    status = "Waiting"
    rects = []

    if totalFrames % args["skip_frames"] == 0:
        status = "Detecting"
        trackers = []


# taking the input frames in size of 416*416
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)
# detection of the vehicles

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > args["confidence"]:
                    box = detection[0:4]*np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")


# starting point of the object
                    x = int(centerX - (width/2))
                    y = int(centerY - (height/2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

# forming the rectangle and tracking the object
        if len(idxs) > 0:
            for i in idxs.flatten():
                if LABELS[classIDs[i]] not in vechiles:
                    continue
                startX = boxes[i][0]
                startY = boxes[i][1]
                endX = boxes[i][0] + boxes[i][2]
                endY = boxes[i][1] + boxes[i][3]

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                trackers.append(tracker)

    else:
        for tracker in trackers:
            status = "Tracking"
            tracker.update(rgb)
            pos = tracker.get_position()
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

# getting the ppm of the video

            ppm = math.sqrt(math.pow(endX-startX, 2))
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            rects.append((startX, startY, endX, endY))

    objects = ct.update(rects)
    speed = 0
    for (objectID, centroid) in objects.items():
        to = trackableOjects.get(objectID, None)

# estimation of the speed
        if to is None:
            to = TrackableObject(objectID, centroid)
        else:
            to.centroids.append(centroid)
            location1 = to.centroids[-2]
            location2 = to.centroids[-1]
            speed = estimateSpeed(location1, location2, ppm, fs)
        trackableOjects[objectID] = to


        cv2.putText(frame, "{:.1f} km/h".format(speed), (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        

    if writer is not None:
        writer.write(frame)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    totalFrames += 1
    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if writer is not None:
	writer.release()

if not args.get("input", False):
	vs.stop()

else:
	vs.release()

cv2.destroyAllWindows()