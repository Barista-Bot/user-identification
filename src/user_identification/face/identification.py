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
        
        self.cv_face_rec = cv2.createLBPHFaceRecognizer(threshold=100)
        self.trained = False
        try:
            if not os.path.exists(self.model_file):
                raise cv2.error
            self.cv_face_rec.load(self.model_file)
            self.trained = True
        except cv2.error:
            pass
           
        if not self.trained:
            faces = []
            IDs = []
            for id in os.listdir(self.raw_face_dir):
                id_path = os.path.join(self.raw_face_dir, id)
                try:
                    id = int(id)
                    id_is_int = True
                except ValueError:
                    id_is_int = False
                if os.path.isdir(id_path) and id_is_int:
                    for img_file in os.listdir(id_path):
                        img_path = os.path.join(id_path, img_file)
                        img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                        if img is not None:
                            faces.append(img)
                            IDs.append(id)
            if len(IDs) > 0:
                self.cv_face_rec.train(faces, np.array(IDs))
                self.trained = True
                self.cv_face_rec.save(self.model_file)
                  
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
            person_id, confidence = self.cv_face_rec.predict(face_img)
            confidence = 100 - confidence
            is_known_person = False if person_id == -1 else True

        return is_known_person, person_id, confidence
