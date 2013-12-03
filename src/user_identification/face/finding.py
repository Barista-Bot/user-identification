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

    @abstractmethod
    def findFacesInImage(self, img):
        """Returns all faces in the image, or None if one doesn't exist"""
        pass


class FaceFinder(AbstractFaceFinder):
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml")
        
    def findFacesInImage(self, img):
        min_face_width = img.shape[1]/10
        min_face = (min_face_width, min_face_width)
        rects = self.face_detector.detectMultiScale(img, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, minSize=min_face)
        rects = [util.Rect(r) for r in rects]
        return rects

    def findLargestFaceInImage(self, img):
        rects = self.findFacesInImage(img)
        return util.largestOfRects(rects)

FaceFinder1 = FaceFinder

import numpy as np
from collections import deque

class MouthFinder(object):
    def __init__(self):
        self._mouth_detector = cv2.CascadeClassifier("/usr/share/opencv/haarcascades/haarcascade_mcs_mouth.xml")
        self._roi_ratio = 0.5
        self._deque_size = 20
        self._openness_history = deque(maxlen=self._deque_size)
        self.mouth_img = None
        self._prev_talkingness = 0

    def getTalkingness(self, frame, face_rect):
        expanded_face_rect = util.Rect([face_rect.pt1.x, face_rect.pt1.y, face_rect.width(), 1.2*face_rect.height()])
        face_img = util.subimage(frame, expanded_face_rect)
        face_mouth_rect = self.findMouthInFace(face_img)
        if face_mouth_rect:
            mouth_img = util.subimage(face_img, face_mouth_rect)
            skin_colour = np.mean([mouth_img[0][0], mouth_img[-1][0], mouth_img[0][1], mouth_img[-1][-1]], 0)
            colour_thresh = 0.35*skin_colour
            openness = 2*(np.sum(mouth_img < colour_thresh)/50)**2

            if self._prev_talkingness < 100 and openness > 300:
                openness *= 10

            self._openness_history.append(openness)
            talkingness = sum(self._openness_history)/len(self._openness_history)

            self.mouth_img = mouth_img
            self._prev_talkingness = talkingness
        return self._prev_talkingness

    def findMouthInFace(self, face_img):
        face_height, face_width, _ = face_img.shape
        mouth_roi_rect = util.Rect([0,face_height*(1-self._roi_ratio), face_width, face_height*self._roi_ratio])
        mouth_roi_img = util.subimage(face_img, mouth_roi_rect)
        rects = self._mouth_detector.detectMultiScale(mouth_roi_img, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE)
        rects = [util.Rect(r) for r in rects]
        rect = util.largestOfRects(rects)
        if rect:
            rect = util.Rect([
                rect.pt1.x + mouth_roi_rect.pt1.x,
                rect.pt1.y + mouth_roi_rect.pt1.y,
                rect.width(),
                rect.height()
            ])
            return rect
        else:
            return None

