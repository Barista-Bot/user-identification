user-identification
====================

ROS node for learning the appearance of people and identifying the current person.

Usage
------

- `rosrun user_identification user_identification_server.py`
- With rosservice:
    - `rosservice call /user_identification/definePerson 0`
     - returns True if successful
    - `rosservice call /user_identification/queryPerson`
     - returns is_person, is_known_person, person_id, confidence
    - `rosservice call /user_identification/exit`
- With Python:
    - `from user_identification import client`
    - `client.definePerson(0)`
    - `client.queryPerson()`


To do
------

- Ability to disable GUI
- Make training data file permanent
- Improve face finding
- Improve face identification
