#!/usr/bin/env python2

import os
import cv2
import numpy as np
from .. import util

class FaceIdentifier(object):
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

    def update(self, face_img, person_id):
        face_img = util.col2bw(face_img)
        self.cv_face_rec.update(np.asarray([face_img]), np.asarray([person_id]))
        self.trained = True
        self.cv_face_rec.save(self.Model_File)

    def predict(self, face_img):
        face_img = util.col2bw(face_img)
        is_known_person, person_id, confidence = False, 0, 0
        if self.trained:
            person_id, confidence = self.cv_face_rec.predict(face_img)
            confidence = 100 - confidence
            if confidence > 10:
                is_known_person = True

        return is_known_person, person_id, confidence