#!/usr/bin/env python

from user_identification.srv import *
from collections import namedtuple
import rospy
import cv2
import numpy as np
import os

class Rect(object):
    Point = namedtuple('Point', 'x y')

    def __init__(self, array):
        x, y, w, h = array

        self.pt1 = self.Point(x, y)
        self.pt2 = self.Point(x + w, y + h)

    def width(self):
        return self.pt2.x - self.pt1.x

    def height(self):
        return self.pt2.y - self.pt1.y

    def area(self):
        return self.width() * self.height()


class FaceTracking(object):
    def __init__(self):
        self.vc = cv2.VideoCapture(0)
        self.face_detector = cv2.CascadeClassifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml")

    def getFaceImage(self): 
        face_rect = None
        while not face_rect:
            frame = self.getFrame()
            face_rect = self.findLargestFaceInImage(frame)
        face_img = self.col2bw(self.subimage(frame, face_rect))
        self.vc.release()
        return face_img

    def getFrame(self):
        rval, frame = self.vc.read()
        if not rval:
            self.exit()
        return frame

    def subimage(self, img, rect):
        return img[rect.pt1.y:rect.pt2.y,  rect.pt1.x:rect.pt2.x]

    def col2bw(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def findFacesInImage(self, img):
        rects = self.face_detector.detectMultiScale(img, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))
        rects = [Rect(r) for r in rects]
        return rects

    def findLargestFaceInImage(self, img):
        rects = self.findFacesInImage(img)
        max_size = 0
        max_rect = None

        for rect in rects:
            size = rect.width()
            if size >= max_size:
                max_size = size
                max_rect = rect

        return max_rect

class FaceRecogniser(object):
    Model_File = '/tmp/face_rec_model'

    def __init__(self):
        self.cv_face_rec = cv2.createLBPHFaceRecognizer()
        try:
            if not os.path.exists(self.Model_File):
                raise cv2.error
            self.cv_face_rec.load(self.Model_File)
            self.trained = True
        except cv2.error:
            self.trained = False

    def predict(self, *args):
        if self.trained:
            person_id, confidence = self.cv_face_rec.predict(*args)
            confidence = 100 - confidence
            return person_id, confidence
        else:
            return None, -1


def queryPerson_handler(obj):
    print "Start tracking..."
    face_tracker = FaceTracking()
    face_recogniser = FaceRecogniser()
    face_img = face_tracker.getFaceImage()
    person_id, confidence = face_recogniser.predict(face_img)
    print ("Person_Id: ", person_id, "; Confidence: ", confidence)
    return queryPersonResponse(person_id, confidence)

def queryPerson_server():
    rospy.init_node('queryPerson_server')
    s = rospy.Service('queryPerson', queryPerson, queryPerson_handler)
    print "Service initiated."
    rospy.spin()

if __name__ == "__main__":
    queryPerson_server() 
