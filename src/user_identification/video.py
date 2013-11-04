#!/usr/bin/env python2

import cv2
import rospy
import numpy as np
from abc import ABCMeta, abstractmethod
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time

DEFAULT_ROS_VIDEO_TOPIC = "/v4l/camera/image_raw"
# DEFAULT_ROS_VIDEO_TOPIC = "/rgb/image_raw"

class AbstractVideoSource(object):
    __metaclass__  = ABCMeta

    @abstractmethod
    def getFrame(self):
        """Returns an image frame from the camera, as a 2D numpy array"""
        pass

class DirectVideoSource(AbstractVideoSource):
    def __init__(self):
        self.vc = cv2.VideoCapture(0)

    def getFrame(self):
        rval, frame = self.vc.read()
        if not rval:
            raise Exception('Video failed')
        return frame

class RosVideoSource1(AbstractVideoSource):
    ros_video_topic = DEFAULT_ROS_VIDEO_TOPIC

    def __init__(self):
        self.frame = None
        self.cvBridge = CvBridge()
        rospy.wait_for_message(self.ros_video_topic, Image)
        self.image_sub = rospy.Subscriber(self.ros_video_topic, Image, self.callback, queue_size=1)

    def callback(self, ros_frame):
        self.frame = np.asarray(self.cvBridge.imgmsg_to_cv(ros_frame, "bgr8"))

    def getFrame(self):
        while self.frame == None:
            time.sleep(0.01)
        return self.frame

class RosVideoSource2(AbstractVideoSource):
    ros_video_topic = DEFAULT_ROS_VIDEO_TOPIC

    def __init__(self):
        self.ros_frame = None
        self.cvBridge = CvBridge()
        rospy.wait_for_message(self.ros_video_topic, Image)
        self.image_sub = rospy.Subscriber(self.ros_video_topic, Image, self.callback, queue_size=1)

    def callback(self, ros_frame):
        self.ros_frame = ros_frame

    def getFrame(self):
        while self.ros_frame == None:
            time.sleep(0.01)
        frame = np.asarray(self.cvBridge.imgmsg_to_cv(self.ros_frame, "bgr8"))
        return frame

class RosVideoSource3(AbstractVideoSource):
    ros_video_topic = DEFAULT_ROS_VIDEO_TOPIC

    def __init__(self):
        self.cvBridge = CvBridge()

    def getFrame(self):
        ros_frame = rospy.wait_for_message(self.ros_video_topic, Image)
        frame = np.asarray(self.cvBridge.imgmsg_to_cv(ros_frame, "bgr8"))
        return frame

