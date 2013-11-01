#!/usr/bin/env python2

import cv2
from abc import ABCMeta, abstractmethod

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
    """To be implemented"""
