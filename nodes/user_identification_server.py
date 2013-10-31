#!/usr/bin/env python2

from collections import namedtuple
import os
import cv2
import numpy as np
import gobject
gobject.threads_init()
import dbus
import dbus.service
import _dbus_bindings as dbus_bindings
from dbus.mainloop.glib import DBusGMainLoop
DBusGMainLoop(set_as_default=True)
import roslib
import rospy
import std_srvs.srv
import importlib

class Util(object):
    @staticmethod    
    def subimage(img, rect):
        return img[rect.pt1.y:rect.pt2.y,  rect.pt1.x:rect.pt2.x]

    @staticmethod        
    def col2bw(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod        
    def drawBoxesOnImage(rects, img):
        for rect in rects:
            cv2.rectangle(img, rect.pt1, rect.pt2, (127, 255, 0), 2)

class Rect(object):
    Point = namedtuple('Point', 'x y')

    def __init__(self, array):
        x, y, w, h = array

        self.pt1 = self.Point(x, y)
        self.pt2 = self.Point(x + w, y + h)

    def width(self):
        return self.pt2.x - self.pt1.x

    def height(self):
        return self.pt2.y - self.pt1.y

    def area(self):
        return self.width() * self.height()

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
        face_img = Util.col2bw(face_img)
        self.cv_face_rec.update(np.asarray([face_img]), np.asarray([person_id]))
        self.trained = True
        self.cv_face_rec.save(self.Model_File)

    def predict(self, face_img):
        face_img = Util.col2bw(face_img)
        is_known_person, person_id, confidence = False, 0, 0
        if self.trained:
            person_id, confidence = self.cv_face_rec.predict(face_img)
            confidence = 100 - confidence
            if confidence > 10:
                is_known_person = True

        return is_known_person, person_id, confidence

class FaceFinder(object):
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml")
        
    def findFacesInImage(self, img):
        rects = self.face_detector.detectMultiScale(img, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))
        rects = [Rect(r) for r in rects]
        return rects

    def findLargestFaceInImage(self, img):
        rects = self.findFacesInImage(img)
        max_size = 0
        max_rect = None

        for rect in rects:
            size = rect.width()
            if size >= max_size:
                max_size = size
                max_rect = rect

        return max_rect

class VideoSource(object):
    def __init__(self):
        self.vc = cv2.VideoCapture(0)

    def getFrame(self):
        rval, frame = self.vc.read()
        if not rval:
            raise Exception('Video failed')
        return frame

        

class UserIdentifierServer(dbus.service.Object):
    Bus_Name = 'org.BaristaBot.user_id'
    Object_Path = '/org/BaristaBot/user_id'
    Interface_Name = 'org.BaristaBot.UserIdInterface'

    PKG_NAME = 'user_identification'
    NODE_NAME = 'user_identification_server'

    def __init__(self, main_loop):
        self.initDbus()
        self.initRosNode()

        self.main_loop = main_loop
        self.vs = VideoSource()
        self.face_identifier = FaceIdentifier()
        self.face_finder = FaceFinder()

    def initDbus(self):
        session = dbus.SessionBus()
        if session.name_has_owner(self.Bus_Name):
            print('This service is already running')
            raise SystemExit(1)
        bus_name = dbus.service.BusName(self.Bus_Name, session)
        super(UserIdentifierServer, self).__init__(bus_name, self.Object_Path)

    def initRosNode(self):
        roslib.load_manifest(self.PKG_NAME)
        self.ros_srv = importlib.import_module(self.PKG_NAME+'.srv')
        rospy.init_node(self.NODE_NAME)

        s = rospy.Service(self.PKG_NAME+'/definePerson', self.ros_srv.definePerson, lambda req: self.definePerson(req.id))
        s = rospy.Service(self.PKG_NAME+'/exit', std_srvs.srv.Empty, lambda req: self.exit())
        s = rospy.Service(self.PKG_NAME+'/queryPerson', self.ros_srv.queryPerson, lambda req: self.queryPerson())
        
    @dbus.service.method(Interface_Name, in_signature='i', out_signature='b')
    def definePerson(self, person_id):
        face_rect = None
        while not face_rect:
            frame = self.vs.getFrame()
            face_rect = self.face_finder.findLargestFaceInImage(frame)
        face_img = Util.subimage(frame, face_rect)
        cv2.imshow("train", face_img)
        self.face_identifier.update(face_img, person_id)

        return True

    @dbus.service.method(Interface_Name, out_signature='bbii')
    def queryPerson(self, ret_rect=False):
        is_person, is_known_person = False, False
        person_id, confidence = 0, 0

        frame = self.vs.getFrame()
        face_rect = self.face_finder.findLargestFaceInImage(frame)
        if face_rect:
            is_person = True
            face_img = Util.subimage(frame, face_rect)
            is_known_person, person_id, confidence = self.face_identifier.predict(face_img)

        if ret_rect:
            return is_person, is_known_person, person_id, confidence, face_rect
        else:
            return is_person, is_known_person, person_id, confidence     

    @dbus.service.method(Interface_Name)
    def exit(self):
        self.main_loop.quit()

    def spinOnce(self):
        frame = self.vs.getFrame()

        is_person, is_known_person, person_id, confidence, face_rect = self.queryPerson(ret_rect=True)
        if is_person:
            Util.drawBoxesOnImage([face_rect], frame)
            if is_known_person:
                label = str(person_id)
                cv2.putText(frame, label+', '+str(confidence), (face_rect.pt1.x, face_rect.pt1.y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (127, 255, 0))

        cv2.imshow("face-id", frame)

        key = cv2.waitKey(20)
        if key == 27:
            self.exit()
        return True

def main():
    main_loop = gobject.MainLoop()
    server = UserIdentifierServer(main_loop)

    gobject.timeout_add(50, server.spinOnce)

    try:
        main_loop.run()
    except KeyboardInterrupt:
        raise SystemExit(0)

if __name__ == "__main__":
    main()