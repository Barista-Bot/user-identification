user-identification
====================

ROS node for learning the appearance of people and identifying the current person.

Usage
------

- `rosrun user_identification user_identification_server.py`
- `rosservice call /user_identification/definePerson 0`
- `rosservice call /user_identification/exit`


To do
------

- Factor out separate classes for individual components
- Expose a queryPerson() method
- Ability to obtain video feed from ROS
- Ability to disable GUI
- Make training data file permanent
- Improve face finding
- Improve face identification