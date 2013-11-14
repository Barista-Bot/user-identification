#!/usr/bin/env python2

import time
import cv2
import numpy as np
from collections import namedtuple
from abc import ABCMeta, abstractmethod
from .. import util

QueryPersonResult = namedtuple('QueryPersonResult', 'is_person is_known_person id confidence face_rect')

class AbstractEngine(object):
    __metaclass__ = ABCMeta

    def __init__(self, face_finder, face_identifier, video_source, publish_method=lambda:None):
        self.face_finder = face_finder
        self.face_identifier = face_identifier
        self.video_source = video_source
        self.publish = publish_method
        self.last_training_image = None

    @abstractmethod
    def definePerson(self):
        pass

    @abstractmethod
    def queryPerson(self):
        pass

    @abstractmethod
    def spinOnce(self):
        pass


class OnDemandEngine(AbstractEngine):
    def definePerson(self, person_id):
        face_rect = None
        while face_rect == None:
            frame = self.video_source.getFrame()
            face_rect = self.face_finder.findLargestFaceInImage(frame)
        face_img = util.subimage(frame, face_rect)
        self.face_identifier.update(face_img, person_id)

        self.last_training_image = face_img
        return True

    def queryPerson(self):
        is_person, is_known_person = False, False
        person_id, confidence = -1, 0

        frame = self.video_source.getFrame()
        face_rect = self.face_finder.findLargestFaceInImage(frame)
        if face_rect:
            is_person = True
            face_img = util.subimage(frame, face_rect)
            is_known_person, person_id, confidence = self.face_identifier.predict(face_img)

        return QueryPersonResult(is_person, is_known_person, person_id, confidence, face_rect)

    def spinOnce(self):
        self.video_source.getNewFrame()
        self.publish()


class ContinuousEngine(AbstractEngine):
    def definePerson(self, person_id):
        while True:
            face_rect = self.face_rect
            if face_rect != None:
                break
            time.sleep(0.01)
        frame = self.video_source.getFrame()
        face_img = util.subimage(frame, face_rect)
        self.face_identifier.update(face_img, person_id)

        self.last_training_image = face_img
        return True

    def queryPerson(self):
        return QueryPersonResult(self.is_person, self.is_known_person, self.person_id, self.confidence, self.face_rect)

    def updatePersonState(self):
        self.is_person, self.is_known_person = False, False
        self.person_id, self.confidence = -1, 0

        frame = self.video_source.getFrame()
        self.face_rect = self.face_finder.findLargestFaceInImage(frame)
        if self.face_rect:
            self.is_person = True
            face_img = util.subimage(frame, self.face_rect)
            self.is_known_person, self.person_id, self.confidence = self.face_identifier.predict(face_img)

    def spinOnce(self):
        self.video_source.getNewFrame()
        self.updatePersonState()
        self.publish()


class ContinuousLKTrackingEngine(AbstractEngine):
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    tracks = []
    track_len = 10
    prev_gray = None
    resetLK = True

    def lkTrack(self):
        frame = self.video_source.getFrame()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if len(self.tracks) > 0:
            img0, img1 = self.prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_tracks = []
            minX = 99999
            minY = 99999
            maxX = -99999
            maxY = -99999
            for tr, (x_tr, y_tr), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x_tr, y_tr))
                if len(tr) > self.track_len:
                    del tr[0]
                new_tracks.append(tr)
                cv2.circle(frame, (x_tr, y_tr), 2, (0, 255, 0), -1)
                minX = min(minX,x_tr)
                minY = min(minY, y_tr)
                maxX = max(maxX, x_tr)
                maxY = max(maxY, y_tr)
            self.tracks = new_tracks
            cv2.polylines(frame, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
            cv2.rectangle(frame,(minX,minY),(maxX,maxY),(255,255,255),2)
        else:
            self.resetLK = True
        self.prev_gray = frame_gray

    def definePerson(self, person_id):
        while True:
            face_rect = self.face_rect
            if face_rect != None:
                break
            time.sleep(0.01)
        frame = self.video_source.getFrame()
        face_img = util.subimage(frame, face_rect)
        self.face_identifier.update(face_img, person_id)
        self.last_training_image = face_img
        return True

    def queryPerson(self):
        return QueryPersonResult(self.is_person, self.is_known_person, self.person_id, self.confidence, self.face_rect)

    def initLK(self):
        self.is_person, self.is_known_person = False, False
        self.person_id, self.confidence = -1, 0

        frame = self.video_source.getFrame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.face_rect = self.face_finder.findLargestFaceInImage(frame)

        if self.face_rect:
            self.is_person = True
            face_img = util.subimage(frame, self.face_rect)
            self.is_known_person, self.person_id, self.confidence = self.face_identifier.predict(face_img)
            mask = np.zeros_like(gray)
            mask[:] = 255
            for mx, my in [np.int32(tr[-1]) for tr in self.tracks]:
                cv2.circle(mask, (mx, my), 5, 0, -1)
            p = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            if p is not None:
                for px, py in np.float32(p).reshape(-1, 2):
                    (point1, point2) = self.face_rect.pt1, self.face_rect.pt2
                    if (point1.x <= px <= point2.x) and (point1.y <= py <= point2.y):
                        self.tracks.append([(px, py)])
                self.resetLK = False
        self.prev_gray = gray.copy()

    def spinOnce(self):
        self.video_source.getNewFrame()
        if self.resetLK:
            self.initLK()
        self.lkTrack()
        self.publish()
