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
        self._flow_history = deque(maxlen=5)
        self._entropy_history = deque(maxlen=6)
        self._prev_talkingness = 0
        self._prev_mouth_img = None

    def getTalkingness(self, frame, face_rect):
        expanded_face_rect = util.Rect((face_rect.pt1.x, face_rect.pt1.y, face_rect.width(), 1.2*face_rect.height()))
        face_img = util.subimage(frame, expanded_face_rect)
        face_mouth_rect = self.findMouthInFace(face_img)
        if face_mouth_rect:
            face_mouth_rect.pt1.y -= 0.1*face_mouth_rect.height()
            face_mouth_rect.pt2.y -= 0.1*face_mouth_rect.height()
            mouth_img = util.subimage(face_img, face_mouth_rect)
            mouth_img = cv2.resize(mouth_img, (116, 70), interpolation=cv2.INTER_AREA)
            mouth_img = cv2.cvtColor(mouth_img,cv2.COLOR_BGR2GRAY)
            # cv2.imshow('mouth', mouth_img)

            if self._prev_mouth_img is not None and np.any(mouth_img != self._prev_mouth_img):
                flow = cv2.calcOpticalFlowFarneback(self._prev_mouth_img,mouth_img, 0.5, 3, 15, 3, 5, 1.2, 0)

                self._flow_history.append(flow)
                flow = np.sum(self._flow_history, axis=0)

                flow_mag, flow_ang = cv2.cartToPolar(flow[...,0], flow[...,1])

                flow_ang_labels = (256*flow_ang/(2*np.pi)).astype('uint8').ravel()
                counts = np.bincount(flow_ang_labels, weights=flow_mag.ravel())
                probs = counts[np.nonzero(counts)].astype(float) / len(flow_ang_labels)

                entropy = - np.sum(probs * np.log(probs))

                if entropy > 90:
                    entropy *= 5

                self._entropy_history.append(entropy)
                entropy = sum(self._entropy_history)/len(self._entropy_history)

                hsv = np.zeros(shape=(flow.shape[0], flow.shape[1], 3), dtype='uint8')
                hsv[...,0] = flow_ang*180/np.pi/2
                hsv[...,1] = 255
                hsv[...,2] = 20*flow_mag # cv2.normalize(flow_mag,None,0,255,cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

                cv2.imshow('flow',rgb)

                self._prev_talkingness = int(entropy*10)

            self._prev_mouth_img = mouth_img
        return self._prev_talkingness

    def findMouthInFace(self, face_img):
        face_height, face_width, _ = face_img.shape
        mouth_roi_rect = util.Rect([0,face_height*(1-self._roi_ratio), face_width, face_height*self._roi_ratio])
        mouth_roi_img = util.subimage(face_img, mouth_roi_rect)
        min_size = tuple(int(0.4*l) for l in mouth_roi_rect.shape())
        max_size = tuple(int(0.6*l) for l in mouth_roi_rect.shape())
        rects = self._mouth_detector.detectMultiScale(mouth_roi_img, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, minSize=min_size, maxSize=max_size)
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

