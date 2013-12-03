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
        cv2.rectangle(img, rect.pt1, rect.pt2, colour, 2)
    return img

def drawPointsOnImage(pts, img):
    for pt in pts:
        cv2.circle(img, pt, 5, (255, 0, 0), -1)
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

    def isPointInRect(self, point):
        if (self.pt1.x <= point.x <= self.pt2.x) and (self.pt1.y <= point.y <= self.pt2.y):
            return True
        return False
