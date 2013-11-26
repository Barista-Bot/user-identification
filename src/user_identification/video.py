#!/usr/bin/env python2

import cv2
import rospy
import numpy as np
from abc import ABCMeta, abstractmethod
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time

DEFAULT_ROS_VIDEO_TOPIC = "/rgb/image_raw"

class AbstractVideoSource(object):
    __metaclass__  = ABCMeta

    @abstractmethod
    def getNewFrame(self):
        """
        Gets a new image frame from the camera as a 2D numpy array, stores it to self.frame and returns it
        """

    def getFrame(self):
        return self.frame

class DirectVideoSource(AbstractVideoSource):
    def __init__(self):
        for i in range(10):
            try:
                self.vc = cv2.VideoCapture(i)
                self.getNewFrame()
                break
            except:
                pass
        else:
            raise Exception('No working video device found between 0 and 9')



    def getNewFrame(self):
        rval, self.frame = self.vc.read()
        if not rval:
            raise Exception('Video failed')
        return self.frame

class RosVideoSource1(AbstractVideoSource):
    ros_video_topic = DEFAULT_ROS_VIDEO_TOPIC

    def __init__(self):
        self.cv_frame = None
        self.frame = None
        self.cvBridge = CvBridge()
        rospy.wait_for_message(self.ros_video_topic, Image)
        self.image_sub = rospy.Subscriber(self.ros_video_topic, Image, self.callback, queue_size=1)

    def callback(self, ros_frame):
        self.cv_frame = np.asarray(self.cvBridge.imgmsg_to_cv(ros_frame, "bgr8"))

    def getNewFrame(self):
        while self.cv_frame == None:
            time.sleep(0.01)
        self.frame = self.cv_frame
        return self.frame

class RosVideoSource2(AbstractVideoSource):
    ros_video_topic = DEFAULT_ROS_VIDEO_TOPIC

    def __init__(self):
        self.ros_frame = None
        self.frame = None
        self.cvBridge = CvBridge()
        rospy.wait_for_message(self.ros_video_topic, Image)
        self.image_sub = rospy.Subscriber(self.ros_video_topic, Image, self.callback, queue_size=1)

    def callback(self, ros_frame):
        self.ros_frame = ros_frame

    def getNewFrame(self):
        while self.ros_frame == None:
            time.sleep(0.01)
        self.frame = np.asarray(self.cvBridge.imgmsg_to_cv(self.ros_frame, "bgr8"))
        return self.frame

class RosVideoSource3(AbstractVideoSource):
    ros_video_topic = DEFAULT_ROS_VIDEO_TOPIC

    def __init__(self):
        self.cvBridge = CvBridge()
        self.frame = None

    def getNewFrame(self):
        ros_frame = rospy.wait_for_message(self.ros_video_topic, Image)
        self.frame = np.asarray(self.cvBridge.imgmsg_to_cv(ros_frame, "bgr8"))
        return self.frame

