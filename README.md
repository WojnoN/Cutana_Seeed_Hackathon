# Team Cutana

Developers of your new chopping kitchen aid.

## Goals
The goal of Cutana is to have a robot who is able to pick up vegetables, place them onto a chopping board and cut them up for you.

## Hardware
1. We used LeRobot SO-101 arms as the main hardware for our project. One arm to execute a tasks and then a second to act as a leader for policy example  trainings.
2. To help the SO-101 complete its tasks, custom equipment was printed to hold the blade, and a mount for the gripper to grab.


## Software
All policies were trained using ACT. 

Instead of trying to record everything in one take, we have decided to split our tasks into smaller policies and train them with smaller recordings.

Each policy implemented was trained after taking 100 episodes, the tasks that were trained being the following:
- Chopping
- Vegetable pickup
- Blade pickup

## How to run
Place cutana.py in your lerobot root (after cloning the repository) and run it with the following command:

python cutana.py   --robot.type=so101_follower   --robot.port=/dev/ttyACM0   --robot.id=follower   --robot.cameras="{robot: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30},
                      top: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}"   --display_data=true   --dataset.repo_id=vednot25t/eval_kitchen2   --dataset.num_episodes=1   --dataset.episode_time_s=15   --dataset.reset_time_s=1   --dataset.single_task="Cut the veg"   --policy_paths='["vednot25t/pickup_carrot","vednot25t/pickup_knife4","vednot25t/cut_veg"]'
