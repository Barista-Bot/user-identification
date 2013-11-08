#!/usr/bin/env python2

import os
import cv2
import gobject
gobject.threads_init()
import dbus
import dbus.service
from dbus.mainloop.glib import DBusGMainLoop
DBusGMainLoop(set_as_default=True)
import roslib
import rospy
import rospkg
import std_srvs.srv
import importlib
from user_identification import util, video, face, gui


class UserIdentifierServer(dbus.service.Object):
    BUS_NAME = 'org.BaristaBot.user_id'
    OBJECT_PATH = '/org/BaristaBot/user_id'
    INTERFACE_NAME = 'org.BaristaBot.UserIdInterface'

    PKG_NAME = 'user_identification'
    NODE_NAME = 'user_identification_server'

    DATA_DIR = os.path.join(rospkg.get_ros_home(), PKG_NAME)

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
        classes = {}
        for i in [
            {'ros_param':'~videosource',    'default_class':'DirectVideoSource' },
            {'ros_param':'~faceidentifier', 'default_class':'FaceIdentifier' },
            {'ros_param':'~facefinder',     'default_class':'FaceFinder1' },
            {'ros_param':'~faceengine',     'default_class':'ContinuousEngine' },
        ]:
            try:
                class_name = rospy.get_param(i['ros_param'])
            except KeyError:
                class_name = i['default_class']
            classes[i['ros_param']] = class_name

        video_source = getattr(video, classes['~videosource'])()
        face_identifier = getattr(face.identification, classes['~faceidentifier'])(data_dir=self.DATA_DIR)
        face_finder = getattr(face.finding, classes['~facefinder'])()

        face_engine_class = getattr(face.engine, classes['~faceengine'])
        self.face_engine = face_engine_class(face_finder, face_identifier, video_source)

    def initDbus(self):
        session = dbus.SessionBus()
        if session.name_has_owner(self.BUS_NAME):
            print('This service is already running')
            raise SystemExit(1)
        bus = dbus.service.BusName(self.BUS_NAME, session)
        super(UserIdentifierServer, self).__init__(bus, self.OBJECT_PATH)

    def initRosNode(self):
        self.ros_srv = importlib.import_module(self.PKG_NAME+'.srv')
        rospy.init_node(self.NODE_NAME)
        rospy.on_shutdown(lambda: self.exit())

        self.rospy_services = [
            rospy.Service(self.PKG_NAME+'/definePerson', self.ros_srv.definePerson, lambda req: self.definePerson(req.id)),
            rospy.Service(self.PKG_NAME+'/queryPerson', self.ros_srv.queryPerson, lambda req: self.queryPerson()),
            rospy.Service(self.PKG_NAME+'/exit', std_srvs.srv.Empty, lambda req: self.exit(from_ros_service=True)),
        ]
  
    @dbus.service.method(INTERFACE_NAME, in_signature='i', out_signature='b')
    def definePerson(self, person_id):
        success, face_img = self.face_engine.definePerson(person_id)
        self.gui.showTrainingImage(face_img)
        return success

    @dbus.service.method(INTERFACE_NAME, out_signature='bbii')
    def queryPerson(self, ret_rect=False):
        is_person, is_known_person, person_id, confidence, rect = self.face_engine.queryPerson()
        return is_person, is_known_person, person_id, confidence

    @dbus.service.method(INTERFACE_NAME)
    def exit(self, from_ros_service=False):
        self.main_loop.quit()
        if from_ros_service:
            return std_srvs.srv.EmptyResponse()

    def spinOnce(self):
        self.face_engine.spinOnce()
        self.gui.spinOnce(self)
        return True


def main():
    main_loop = gobject.MainLoop()
    server = UserIdentifierServer(main_loop)

    gobject.timeout_add(20, server.spinOnce)

    try:
        main_loop.run()
    except KeyboardInterrupt:
        main_loop.quit()
        raise SystemExit(0)

if __name__ == "__main__":
    main()