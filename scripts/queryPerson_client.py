#!/usr/bin/env python
import roslib;
import sys
import rospy
from user_identification.srv import *

def queryPerson_client():
    rospy.wait_for_service('queryPerson')
    try:
        queryPerson_service = rospy.ServiceProxy('queryPerson', queryPerson)
        resp = queryPerson_service()
        return resp
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

if __name__ == "__main__":
    print "Querying Person..."
    person = queryPerson_client();
    print "Person_Id: %s; Confidence: %s"%(person.person_id, person.confidence)
