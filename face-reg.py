#!/usr/bin/env python2

import cv2
import numpy as np

def detect(img):
    cascade = cv2.CascadeClassifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml")
    rects = cascade.detectMultiScale(img, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))

    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects

def draw_boxes(rects, img):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)

def main():
    vc = cv2.VideoCapture(0)
    cv2.namedWindow("face-id")

    while True:
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if not rval or key == 27:
            break
        rects = detect(frame)
        draw_boxes(rects, frame)
        cv2.imshow("face-id", frame)

if __name__ == "__main__":
    main()