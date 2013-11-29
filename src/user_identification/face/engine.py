#!/usr/bin/env python2

import time
import cv2
import numpy as np
from collections import namedtuple, Counter, deque
from itertools import islice
from abc import ABCMeta, abstractmethod
from threading import Lock
from .. import util

QueryPersonResult = namedtuple('QueryPersonResult', 'is_person is_known_person id confidence face_rect')

class AbstractEngine(object):
    __metaclass__ = ABCMeta

    def __init__(self, face_finder, face_identifier, video_source, publish_method=lambda:None):
        self._face_finder = face_finder
        self._face_identifier = face_identifier
        self._video_source = video_source
        self._publish = publish_method
        self._last_training_image = None
        self._face_img = None
        self._face_rect = None

    def getFrame(self):
        return self._video_source.getFrame()

    def getLastTrainingImage(self):
        return self._last_training_image
        
    def getTrackingPts(self):
        return []

    @abstractmethod
    def definePerson(self, person_id, is_aborted_method=lambda: False):
        pass

    @abstractmethod
    def queryPerson(self):
        pass

    @abstractmethod
    def spinOnce(self):
        pass

class AveragingEngine(AbstractEngine):
    def __init__(self, inner_engine):
        self._inner_engine = inner_engine
        self._deque_size = 500
        self._person_history = deque(maxlen=self._deque_size)
        self._prev_is_person = False
        self._queryPersonLock = Lock()

    def queryPerson(self):
        self._queryPersonLock.acquire()

        person = self._inner_engine.queryPerson()
        self._person_history.appendleft(person)

        n_samples = 500 if self._prev_is_person else 10

        is_person_counter = Counter(p.is_person for p in islice(self._person_history, 0, n_samples))
        is_person = is_person_counter.most_common(1)[0][0]

        person_id_counter = Counter(p.id for p in islice(self._person_history, 0, n_samples))
        person_id = person_id_counter.most_common(1)[0][0]

        if person.is_person:
            self._last_confidence = person.confidence
            self._last_rect = person.face_rect

        if is_person and not self._prev_is_person:
            self._person_history.extendleft([person]*self._deque_size)

        if not is_person:
            result = QueryPersonResult(is_person, False, -1, 0, None)
        else:

            for p in self._person_history:
                if p.is_person:
                    last_person = p
                    break

            result = QueryPersonResult(is_person, person_id != -1, person_id, self._last_confidence, self._last_rect)

        self._prev_is_person = is_person

        self._queryPersonLock.release()
        return result


    def definePerson(self, person_id, is_aborted_method=lambda: False):
        return self._inner_engine.definePerson(person_id, is_aborted_method)

    def getFrame(self):
        return self._inner_engine._video_source.getFrame()

    def getLastTrainingImage(self):
        return self._inner_engine._last_training_image
       
    def getTrackingPts(self):
        return self._inner_engine.getTrackingPts()

    def spinOnce(self):
        self._inner_engine.spinOnce()



class OnDemandEngine(AbstractEngine):
    def definePerson(self, person_id, is_aborted_method=lambda: False):
        face_rect = None
        while face_rect == None:
            if is_aborted_method():
                return False
            frame = self._video_source.getFrame()
            face_rect = self._face_finder.findLargestFaceInImage(frame)
        face_img = util.subimage(frame, face_rect)
        self._face_identifier.update(face_img, person_id)

        self._last_training_image = face_img
        return True

    def queryPerson(self):
        is_person, is_known_person = False, False
        person_id, confidence = -1, 0

        frame = self._video_source.getFrame()
        face_rect = self._face_finder.findLargestFaceInImage(frame)
        if face_rect:
            is_person = True
            face_img = util.subimage(frame, face_rect)
            is_known_person, person_id, confidence = self._face_identifier.predict(face_img)

        return QueryPersonResult(is_person, is_known_person, person_id, confidence, face_rect)

    def spinOnce(self):
        self._video_source.getNewFrame()
        self._publish()


class ContinuousEngine(AbstractEngine):
    def definePerson(self, person_id, is_aborted_method=lambda: False):
        while True:
            if is_aborted_method():
                return False
            face_img = self._face_img
            if face_img != None and face_img is not self._last_training_image:
                break
            time.sleep(0.01)
        self._face_identifier.update(face_img, person_id)
        self._last_training_image = face_img
        return True

    def queryPerson(self):
        return QueryPersonResult(self.is_person, self.is_known_person, self.person_id, self.confidence, self._face_rect)

    def updatePersonState(self):
        self.is_person, self.is_known_person = False, False
        self.person_id, self.confidence = -1, 0

        frame = self._video_source.getFrame()
        self._face_rect = self._face_finder.findLargestFaceInImage(frame)
        if self._face_rect:
            self.is_person = True
            self._face_img = util.subimage(frame, self._face_rect)
            self.is_known_person, self.person_id, self.confidence = self._face_identifier.predict(self._face_img)

    def spinOnce(self):
        self._video_source.getNewFrame()
        self.updatePersonState()
        self._publish()


class ContinuousLKTrackingEngine(AbstractEngine):
    feature_params = dict( maxCorners = 200,
                           qualityLevel = 0.01,
                           minDistance = 7,
                           blockSize = 7 )

    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    tracks = []
    tracking_pts_to_draw = []
    track_len = 10
    prev_gray = None
    resetLK = True
    avgTrackPoint = util.Rect.Point(0, 0)

    def lkTrack(self):
        frame = self._video_source.getFrame()
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
            sumX = 0
            sumY = 0
            self.tracking_pts_to_draw = []
            for tr, (x_tr, y_tr), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x_tr, y_tr))
                if len(tr) > self.track_len:
                    del tr[0]
                new_tracks.append(tr)
                #cv2.circle(frame, (x_tr, y_tr), 2, (0, 255, 0), -1)
                minX = min(minX,x_tr)
                minY = min(minY, y_tr)
                maxX = max(maxX, x_tr)
                maxY = max(maxY, y_tr)
                sumX += x_tr
                sumY += y_tr
                self.tracking_pts_to_draw += [(x_tr, y_tr)]
            self.avgTrackPoint = util.Rect.Point(int(sumX / len(self.tracks)), int(sumY / len(self.tracks)))
            #cv2.circle(frame, self.avgTrackPoint, 5, (255, 255, 255), -1)
            self.tracks = new_tracks
            #cv2.polylines(frame, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
            #cv2.rectangle(frame,(minX,minY),(maxX,maxY),(255,255,255),2)
        else:
            self.resetLK = True
        self.prev_gray = frame_gray

    def definePerson(self, person_id, is_aborted_method=lambda: False):
        while True:
            if is_aborted_method():
                return False
            face_img = self._face_img
            if face_img != None and face_img is not self._last_training_image:
                break
            time.sleep(0.01)
        self._face_identifier.update(face_img, person_id)
        self._last_training_image = face_img
        return True

    def queryPerson(self):
        return QueryPersonResult(self.is_person, self.is_known_person, self.person_id, self.confidence, self._face_rect)

    def initLK(self):
        self.is_person, self.is_known_person = False, False
        self.person_id, self.confidence = -1, 0
        self.tracks = []

        frame = self._video_source.getFrame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._face_rect = self._face_finder.findLargestFaceInImage(frame)

        if self._face_rect:
            self.is_person = True
            face_img = util.subimage(frame, self._face_rect)
            self.is_known_person, self.person_id, self.confidence = self._face_identifier.predict(face_img)
            mask = np.zeros_like(gray)
            mask[:] = 255
            for mx, my in [np.int32(tr[-1]) for tr in self.tracks]:
                cv2.circle(mask, (mx, my), 5, 0, -1)
            p = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            if p is not None:
                (point1, point2) = self._face_rect.pt1, self._face_rect.pt2
                ptScaleX = (int(point2.x - point1.x) * 0)
                ptScaleY = (int(point2.y - point1.y) * 0)
                for px, py in np.float32(p).reshape(-1, 2):
                    if (point1.x + ptScaleX <= px <= point2.x - ptScaleX) and (point1.y + ptScaleY <= py <= point2.y - ptScaleY):
                        self.tracks.append([(px, py)])
                self.resetLK = False
        self.prev_gray = gray.copy()

    def updatePersonState(self):
        frame = self._video_source.getFrame()
        face_rects = self._face_finder.findFacesInImage(frame)
        if len(face_rects) > 0:
            for face in face_rects:
                if face.isPointInRect(self.avgTrackPoint):
                    self._face_rect = face
                    self._face_img = util.subimage(frame, self._face_rect)
                    self.is_known_person, self.person_id, self.confidence = self._face_identifier.predict(self._face_img)
                    return
            self.resetLK = True
                        
    def getTrackingPts(self):
        return self.tracking_pts_to_draw

    def spinOnce(self):
        self._video_source.getNewFrame()
        if self.resetLK:
            self.initLK()
        self.lkTrack()
        self.updatePersonState()
        self._publish()
