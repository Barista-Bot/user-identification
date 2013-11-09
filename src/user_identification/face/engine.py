#!/usr/bin/env python2

import time
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
