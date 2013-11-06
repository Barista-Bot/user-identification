#!/usr/bin/env python2

import cv2
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
from user_identification import util, video, face, gui

class UserIdentifierServer(dbus.service.Object):
    Bus_Name = 'org.BaristaBot.user_id'
    Object_Path = '/org/BaristaBot/user_id'
    Interface_Name = 'org.BaristaBot.UserIdInterface'

    PKG_NAME = 'user_identification'
    NODE_NAME = 'user_identification_server'

    def __init__(self, main_loop):
        self.main_loop = main_loop
        self.initDbus()
        self.initRosNode()
        self.getConfiguration()

    def getConfiguration(self):
        # Enable/disable GUI
        try:
            use_gui = rospy.get_param('~gui')
        except KeyError:
            use_gui = False
        self.gui = gui.Gui() if use_gui else gui.NoGui()

        # Get other classes to be used
        for i in [
            {'store_to':'vs',              'ros_param':'~videosource',    'default_class':'DirectVideoSource', 'from_module':video},
            {'store_to':'face_identifier', 'ros_param':'~faceidentifier', 'default_class':'FaceIdentifier',    'from_module':face.identification},
            {'store_to':'face_finder',     'ros_param':'~facefinder',     'default_class':'FaceFinder1',       'from_module':face.finding},
        ]:
            try:
                class_name = rospy.get_param(i['ros_param'])
            except KeyError:
                class_name = i['default_class']
            setattr(self, i['store_to'], getattr(i['from_module'], class_name)())


    def initDbus(self):
        session = dbus.SessionBus()
        if session.name_has_owner(self.Bus_Name):
            print('This service is already running')
            raise SystemExit(1)
        bus_name = dbus.service.BusName(self.Bus_Name, session)
        super(UserIdentifierServer, self).__init__(bus_name, self.Object_Path)

    def initRosNode(self):
        self.ros_srv = importlib.import_module(self.PKG_NAME+'.srv')
        rospy.init_node(self.NODE_NAME)
        rospy.on_shutdown(lambda: self.exit())

        self.rospy_services = [
            rospy.Service(self.PKG_NAME+'/definePerson', self.ros_srv.definePerson, lambda req: self.definePerson(req.id)),
            rospy.Service(self.PKG_NAME+'/queryPerson', self.ros_srv.queryPerson, lambda req: self.queryPerson()),
            rospy.Service(self.PKG_NAME+'/exit', std_srvs.srv.Empty, lambda req: self.exit(from_ros_service=True)),
        ]
  
    @dbus.service.method(Interface_Name, in_signature='i', out_signature='b')
    def definePerson(self, person_id):
        face_rect = None
        while not face_rect:
            frame = self.vs.getFrame()
            face_rect = self.face_finder.findLargestFaceInImage(frame)
        face_img = util.subimage(frame, face_rect)
        self.gui.showTrainingImage(face_img)
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
            face_img = util.subimage(frame, face_rect)
            is_known_person, person_id, confidence = self.face_identifier.predict(face_img)

        if ret_rect:
            return is_person, is_known_person, person_id, confidence, face_rect
        else:
            return is_person, is_known_person, person_id, confidence     

    @dbus.service.method(Interface_Name)
    def exit(self, from_ros_service=False):
        self.main_loop.quit()
        if from_ros_service:
            return std_srvs.srv.EmptyResponse()

    def spinOnce(self):
        self.face_finder.spinOnce()
        self.gui.spinOnce(self)
        return True


def main():
    main_loop = gobject.MainLoop()
    server = UserIdentifierServer(main_loop)

    gobject.timeout_add(50, server.spinOnce)

    try:
        main_loop.run()
    except KeyboardInterrupt:
        main_loop.quit()
        raise SystemExit(0)

if __name__ == "__main__":
    main()