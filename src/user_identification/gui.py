#!/usr/bin/env python2

import cv2
import numpy as np
import threading
import util

class Gui(object):
    def __init__(self, server):
        self.server = server
        self.video_publish_thread = None

    def spinOnce(self):
        is_person, is_known_person, person_id, confidence, face_rect, talkingness = self.server.face_engine.queryPerson()
        frame = self.server.face_engine.getFrame()

        if talkingness > 50:
            box_colour = (0, 0, 255)
        else:
            box_colour = (127, 255, 0)

        if is_person:
        
            frame = np.copy(frame)
            util.drawBoxesOnImage([face_rect], frame, box_colour)
            
            tracking_pts = self.server.face_engine.getTrackingPts()
            util.drawPointsOnImage(tracking_pts, frame)
                            
            if is_known_person:
                label = str(person_id)
                cv2.putText(frame, label+', '+str(confidence), (face_rect.pt1.x, face_rect.pt1.y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (127, 255, 0))
        

        cv2.imshow("User Identification", frame)

        if not self.video_publish_thread or not self.video_publish_thread.is_alive():
            self.video_publish_thread = threading.Thread(target=self.server.videoPublish, args=(frame,))
            self.video_publish_thread.start()

        training_img = self.server.face_engine.getLastTrainingImage()
        if training_img != None:
            cv2.imshow("Training Image", training_img)
            
        key = cv2.waitKey(20)
        if key == 27:
            self.server.exit()


class NoGui(object):
    def __init__(self, *args):
        pass

    def spinOnce(self):
        pass
