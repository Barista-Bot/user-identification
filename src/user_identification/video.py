#!/usr/bin/env python2

import cv2
import rospy
import numpy
from abc import ABCMeta, abstractmethod
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

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

class RosVideoSource(AbstractVideoSource):
    rgb_image_path = "/rgb/image_raw"
    frame = None
    frame_set = None

    def __init__(self):
        rospy.wait_for_message(self.rgb_image_path, Image)
        self.cvBridge = CvBridge()
        self.image_sub = rospy.Subscriber(self.rgb_image_path, Image, self.callback, queue_size=1)

    def callback(self, data):
        try:
            frame_mat = self.cvBridge.imgmsg_to_cv(data, "bgr8")
            self.frame = numpy.asarray(frame_mat)
            self.frame_set = True
        except CvBridgeError, e:
            self.frame_set = False
            print e
            raise Exception('RosVideo Failed')

    def getFrame(self):
        while not self.frame_set:
            print "Waiting for frame..."
        return self.frame
