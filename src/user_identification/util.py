#!/usr/bin/env python2

import cv2
import numpy as np
from collections import namedtuple

def subimage(img, rect):
    return img[rect.pt1.y:rect.pt2.y,  rect.pt1.x:rect.pt2.x]

def col2bw(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def drawBoxesOnImage(rects, img, colour=(127, 255, 0)):
    for rect in rects:
        cv2.rectangle(img, tuple(rect.pt1), tuple(rect.pt2), colour, 2)
    return img

def drawPointsOnImage(pts, img):
    for pt in pts:
        cv2.circle(img, tuple(pt), 5, (255, 0, 0), -1)
    return img

def largestOfRects(rects):
    max_size = 0
    max_rect = None

    for rect in rects:
        size = rect.width()
        if size >= max_size:
            max_size = size
            max_rect = rect
    return max_rect

class Point(object):
    def __init__(self, x=None, y=None, pos=None):
        if pos is not None:
            self.pos = pos
        elif x is not None and y is not None:
            self.pos = np.array([x, y])
        else:
            raise Exception()

    def x():
        def fget(self):
            return self.pos[0]
        def fset(self, value):
            self.pos[0] = value
        return locals()
    x = property(**x())

    def y():
        def fget(self):
            return self.pos[1]
        def fset(self, value):
            return self.pos[1]
        return locals()
    y = property(**y())

    def __repr__(self):
        return '(' + str(self.x) + ',' + str(self.y) + ')'

    def __add__(self, x):
        return Point(pos=self.pos + x.pos)

    def __sub__(self, x):
        return Point(pos=self.pos - x.pos)

    def __mul__(self, x):
        return Point(pos=self.pos * x)

    def average(self, x):
        return Point(pos=np.mean([list(self),list(x)], axis=0))

    def __iter__(self):
        yield self.x
        yield self.y

class Rect(object):
    Point = Point

    def __init__(self, array=None, pt1=None, pt2=None):
        if array is not None:
            x, y, w, h = array

            self.pt1 = self.Point(x, y)
            self.pt2 = self.Point(x + w, y + h)
        elif pt1 is not None and pt2 is not None:
            self.pt1 = pt1
            self.pt2 = pt2
        else:
            raise Exception()

    def width(self):
        return self.pt2.x - self.pt1.x

    def height(self):
        return self.pt2.y - self.pt1.y

    def area(self):
        return self.width() * self.height()

    def isPointInRect(self, point):
        if (self.pt1.x <= point.x <= self.pt2.x) and (self.pt1.y <= point.y <= self.pt2.y):
            return True
        return False

    def shape(self):
        return (self.width(), self.height())

    def __mul__(self, x):
        pt_avg = self.pt1.average(self.pt2)
        vec1 = self.pt1 - pt_avg
        vec2 = self.pt2 - pt_avg
        return Rect(pt1=pt_avg+vec1*x, pt2=pt_avg+vec2*x)

    def __rmul__(self, x):
        return self * x




