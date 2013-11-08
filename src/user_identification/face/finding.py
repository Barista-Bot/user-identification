#!/usr/bin/env python2

import cv2
from abc import ABCMeta, abstractmethod
from .. import util


class AbstractFaceFinder(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def findLargestFaceInImage(self, img):
        """Returns a rectangle for the largest face in the image, or None if one doesn't exist"""
        pass


class FaceFinder1(AbstractFaceFinder):
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml")
        
    def findFacesInImage(self, img):
        rects = self.face_detector.detectMultiScale(img, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))
        rects = [util.Rect(r) for r in rects]
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
