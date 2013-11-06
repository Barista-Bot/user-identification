#!/usr/bin/env python2

import rospy
from collections import namedtuple
from user_identification import srv

PKG_NAME = 'user_identification'
DEFINE_PERSON_SERVICE_NAME = PKG_NAME+'/definePerson'
QUERY_PERSON_SERVICE_NAME = PKG_NAME+'/queryPerson'

QueryPersonResult = namedtuple('QueryPersonResult', 'is_person is_known_person id confidence')

def definePerson(person_id):
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
