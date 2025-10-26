## Team Cutana

Developers of your new chopping kitchen aid.

# Goals
The goal of Cutana is to have a robot who is able to pick up vegetables, place them onto a chopping board and cut them up for you.

# Hardware
1. We used LeRobot SO-101 arms as the main hardware for our project. One arm to execute a tasks and then a second to act as a leader for policy example  trainings.
2. To help the SO-101 complete its tasks, custom equipment was printed to hold the blade, and a mount for the gripper to grab.


# Software
All policies were instead trained using ACT.

Instead of trying to record everything in one take, we have decided to split our tasks into smaller policies and train them with smaller recordings.

Each policy implemented was trained after taking 100 episodes, the tasks that were trained being the following:
- Chopping
- Vegetable pickup
- Blade pickup
