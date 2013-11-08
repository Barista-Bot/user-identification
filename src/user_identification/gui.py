#!/usr/bin/env python2

import cv2
import util


class Gui(object):
    def spinOnce(self, server):
        is_person, is_known_person, person_id, confidence, face_rect = server.face_engine.queryPerson()
        frame = server.face_engine.video_source.getFrame()

        if is_person:
            frame = util.drawBoxesOnImage([face_rect], frame)
            if is_known_person:
                label = str(person_id)
                cv2.putText(frame, label+', '+str(confidence), (face_rect.pt1.x, face_rect.pt1.y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (127, 255, 0))

        cv2.imshow("face-id", frame)

        key = cv2.waitKey(20)
        if key == 27:
            server.exit()

    def showTrainingImage(self, img):
        cv2.imshow("train", img)


class NoGui(object):
    def spinOnce(self, server):
        pass

    def showTrainingImage(self, img):
        pass
