#!/usr/bin/env python2

import cv2
from collections import namedtuple

def subimage(img, rect):
    return img[rect.pt1.y:rect.pt2.y,  rect.pt1.x:rect.pt2.x]

def col2bw(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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