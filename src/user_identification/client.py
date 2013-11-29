#!/usr/bin/env python2

import rospy
import actionlib
from collections import namedtuple
from user_identification import srv
from user_identification import msg

PKG_NAME = 'user_identification'
DEFINE_PERSON_SERVICE_NAME = PKG_NAME+'/definePerson'
QUERY_PERSON_SERVICE_NAME = PKG_NAME+'/queryPerson'
PERSON_PRESENCE_TOPIC_NAME = PKG_NAME+'/presence'

QueryPersonResult = namedtuple('QueryPersonResult', 'is_person is_known_person id confidence')

def definePerson(person_id):
    client = actionlib.SimpleActionClient(DEFINE_PERSON_SERVICE_NAME, msg.definePersonAction)
    client.wait_for_server()
    client.send_goal(msg.definePersonGoal(person_id))
    return client

def definePersonService(person_id):
    rospy.wait_for_service(DEFINE_PERSON_SERVICE_NAME)
    func = rospy.ServiceProxy(DEFINE_PERSON_SERVICE_NAME, srv.definePerson)
    result = func(person_id)
    return result.ret

def queryPerson():
    rospy.wait_for_service(QUERY_PERSON_SERVICE_NAME)
    func = rospy.ServiceProxy(QUERY_PERSON_SERVICE_NAME, srv.queryPerson)
    res = func()
    return QueryPersonResult(   
        is_person=res.is_person,
        is_known_person=res.is_known_person,
        id=res.id,
        confidence=res.confidence
    )

def subscribe(callback):
    rospy.wait_for_message(PERSON_PRESENCE_TOPIC_NAME, msg.presence)
    rospy.Subscriber(PERSON_PRESENCE_TOPIC_NAME, msg.presence, callback, queue_size=1)