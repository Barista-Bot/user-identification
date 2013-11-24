#!/usr/bin/env python2

import os
import cv2
import numpy as np
from datetime import datetime
from .. import util

class LBPHIdentifier(object):
    def __init__(self, data_dir='/tmp/user_identification'):
        self.raw_face_dir = os.path.join(data_dir, 'raw_faces')
        try: 
            os.makedirs(self.raw_face_dir)
        except OSError:
            if not os.path.isdir(self.raw_face_dir):
                raise
        
        self.model_file = os.path.join(data_dir, 'face_rec_model')
        
        self.cv_face_rec = cv2.createLBPHFaceRecognizer()
        try:
            if not os.path.exists(self.model_file):
                raise cv2.error
            self.cv_face_rec.load(self.model_file)
            self.trained = True
        except cv2.error:
            self.trained = False

    def update(self, face_img, person_id):
        face_img = util.col2bw(face_img)
        self.cv_face_rec.update(np.asarray([face_img]), np.asarray([person_id]))
        self.trained = True
        self.cv_face_rec.save(self.model_file)

        id_dir = os.path.join(self.raw_face_dir, str(person_id))
        try: 
            os.makedirs(id_dir)
        except OSError:
            if not os.path.isdir(id_dir):
                raise
        cv2.imwrite(os.path.join(id_dir, datetime.now().isoformat()+'.png'), face_img)

    def predict(self, face_img):
        face_img = util.col2bw(face_img)
        is_known_person, person_id, confidence = False, -1, 0
        if self.trained:
            pid, confidence = self.cv_face_rec.predict(face_img)
            confidence = 100 - confidence
            if confidence > 10:
                is_known_person = True
                person_id = pid

        return is_known_person, person_id, confidence