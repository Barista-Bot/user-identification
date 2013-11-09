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
        params = {}
        for i in [
            {'param_name':'~videosource',    'default_val':'DirectVideoSource' },
            {'param_name':'~faceidentifier', 'default_val':'FaceIdentifier' },
            {'param_name':'~facefinder',     'default_val':'FaceFinder1' },
            {'param_name':'~faceengine',     'default_val':'ContinuousEngine' },
            {'param_name':'~gui',            'default_val': True },
        ]:
            try:
                param_val = rospy.get_param(i['param_name'])
            except KeyError:
                param_val = i['default_val']
                rospy.set_param(i['param_name'], param_val)
            params[i['param_name']] = param_val

        self.gui = gui.Gui(self) if params['~gui'] else gui.NoGui()

        video_source = getattr(video, params['~videosource'])()
        face_identifier = getattr(face.identification, params['~faceidentifier'])(data_dir=self.DATA_DIR)
        face_finder = getattr(face.finding, params['~facefinder'])()
        face_engine_class = getattr(face.engine, params['~faceengine'])
        self.face_engine = face_engine_class(face_finder, face_identifier, video_source, self.rosPublish)

    def initDbus(self):
        session = dbus.SessionBus()
        if session.name_has_owner(self.BUS_NAME):
            print('This service is already running')
            raise SystemExit(1)
        bus = dbus.service.BusName(self.BUS_NAME, session)
        super(UserIdentifierServer, self).__init__(bus, self.OBJECT_PATH)

    def initRosNode(self):
        self.ros_srv = importlib.import_module(self.PKG_NAME+'.srv')
        self.ros_msg = importlib.import_module(self.PKG_NAME+'.msg')
        rospy.init_node(self.NODE_NAME)
        rospy.on_shutdown(lambda: self.exit())

        self.ros_services = [
            rospy.Service(self.PKG_NAME+'/definePerson', self.ros_srv.definePerson, lambda req: self.definePerson(req.id)),
            rospy.Service(self.PKG_NAME+'/queryPerson', self.ros_srv.queryPerson, lambda req: self.queryPerson()),
            rospy.Service(self.PKG_NAME+'/exit', std_srvs.srv.Empty, lambda req: self.exit(from_ros_service=True)),
        ]

        self.ros_publisher = rospy.Publisher(self.PKG_NAME+'/presence', self.ros_msg.presence)

  
    @dbus.service.method(INTERFACE_NAME, in_signature='i', out_signature='b')
    def definePerson(self, person_id):
        success = self.face_engine.definePerson(person_id)
        return success

    @dbus.service.method(INTERFACE_NAME, out_signature='bbii')
    def queryPerson(self, ret_rect=False):
        res = self.face_engine.queryPerson()
        return res.is_person, res.is_known_person, res.id, res.confidence

    @dbus.service.method(INTERFACE_NAME)
    def exit(self, from_ros_service=False):
        self.main_loop.quit()
        if from_ros_service:
            return std_srvs.srv.EmptyResponse()

    def rosPublish(self):
        res = self.face_engine.queryPerson()
        msg = self.ros_msg.presence(res.is_person, res.is_known_person, res.id, res.confidence)
        rospy.loginfo(msg)
        self.ros_publisher.publish(msg)

    def spinOnce(self):
        self.face_engine.spinOnce()
        self.gui.spinOnce()
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