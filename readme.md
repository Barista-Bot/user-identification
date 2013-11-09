user-identification
====================

ROS node for learning the appearance of people and identifying the current person.

Usage
------

- Start server:
    - In normal mode: (no GUI, get video from ROS topic)
        - `roslaunch user_identification main.launch`
    - In demo mode: (GUI, connect directly to webcam)
        - `roslaunch user_identification demo.launch`
- Interact:
    - From command-line:
        - `rosservice call /user_identification/definePerson 0`
            - returns True if successful
        - `rosservice call /user_identification/queryPerson`
            - returns is_person, is_known_person, person_id, confidence
        - `rosservice call /user_identification/exit`
    - In Python:
        - `from user_identification import client`
        - `client.definePerson(0)`
        - `client.queryPerson()`
        - Return values are the same as for the command line
