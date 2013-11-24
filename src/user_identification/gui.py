#!/usr/bin/env python2

import cv2
import util


class Gui(object):
    def __init__(self, server):
        self.server = server

    def spinOnce(self):
        is_person, is_known_person, person_id, confidence, face_rect = self.server.face_engine.queryPerson()
        frame = self.server.face_engine.getFrame()

        if is_person:
            frame = util.drawBoxesOnImage([face_rect], frame)
            if is_known_person:
                label = str(person_id)
                cv2.putText(frame, label+', '+str(confidence), (face_rect.pt1.x, face_rect.pt1.y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (127, 255, 0))

        cv2.imshow("User Identification", frame)

        training_img = self.server.face_engine.getLastTrainingImage()
        if training_img != None:
            cv2.imshow("Training Image", training_img)

        key = cv2.waitKey(20)
        if key == 27:
            self.server.exit()


class NoGui(object):
    def spinOnce(self):
        pass
