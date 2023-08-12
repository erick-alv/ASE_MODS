# ASE_MODS

Here we provide our reimplementation of the method presented in "QuestSim: Human Motion Tracking from Sparse Sensors with Simulated Avatars" (https://arxiv.org/abs/2209.09391). We base our code on  the repository of the authors of
"ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters" (https://xbpeng.github.io/projects/ASE/index.html). 


# Requirements
- We used Ubuntu 20.04 for development.
- Nvidia GPU. Installation of the respective drivers and Cuda.
- [Anaconda](https://www.anaconda.com/)
- We installed [Isaac Gym](https://developer.nvidia.com/isaac-gym) and create the corresponding conda environment as described in its' documentation.

# Installation
After installing Isaac Gym and activating the conda environment, go to the root of this repository and run
```
pip install -r requirements.txt
```

# Dataset
For the dataset we have two type of files. The files with the actual motion data and yaml files with filenames. The yaml files represent the data splits that we used.
Download the motion files from https://syncandshare.lrz.de/getlink/fi6vrL9FpeSM2hbzySDrnw/datasets_files and the yaml files from https://syncandshare.lrz.de/getlink/fiV78B2QVEy1hhxKRFvxPk/datasets_yamls.zip.
Please decompress all zip files in the folder ase/data/motions.

# Network weigths
The network weights can be found at https://syncandshare.lrz.de/getlink/fi6vrL9FpeSM2hbzySDrnw/datasets_files . For a description, please read https://gitlab.lrz.de/ga74kob/rl-charactercontrol/-/blob/main/training_output/README.md .

# Training commands
Activate the conda environment and then run
```
python ase/run.py --task HumanoidImitationTrack --cfg_env <path-to-env-config> --cfg_train <path-to-train-config> --motion_file <path-to-the-motion-dataset-file> --headless
```
If desired, you can omit the `--headless` flag to render the environment. Nonetheless, this will require more time to train.

For `<path-to-env-config>` you can use:
- ase/data/cfg/humanoid_imitation_vrh.yaml for torques, r<sub>t,1</sub>, 36Hz and tracker setup H+2C
- ase/data/cfg/humanoid_imitation_vrh_pd.yaml for pd-control, r<sub>t,1</sub>, 36Hz and tracker setup H+2C
- ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml for pd-control, r<sub>t,2</sub>, 36Hz and tracker setup H+2C
- ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml for torques, r<sub>t,2</sub>, 36Hz and tracker setup H+2C
- ase/data/cfg/humanoid_imitation_vrhm2Five.yaml for torques, r<sub>t,1</sub>, 36Hz and tracker setup H+4C
- ase/data/cfg/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml for pd-control, r<sub>t,2</sub>, 36Hz and tracker setup H+4C
- ase/data/cfg/humanoid_imitation_vrhm2Five_rewPenaltyAndReach.yaml for torques, r<sub>t,2</sub>, 36Hz and tracker setup H+4C
- ase/data/cfg/humanoid_imitation_vrhOne.yaml for torques, r<sub>t,1</sub>, 36Hz and tracker setup H
- ase/data/cfg/humanoid_imitation_vrhOne_pd_rewPenaltyAndReach.yaml for pd-control, r<sub>t,2</sub>, 36Hz and tracker setup H
- ase/data/cfg/humanoid_imitation_vrhOne_rewPenaltyAndReach.yaml for torques, r<sub>t,2</sub>, 36Hz and tracker setup H
- Furthermore under ase/data/cfg/other_setup/ you find the same configs but for 72Hz.

For `<path-to-train-config>` you can use:
- ase/data/cfg/train/rlg/common_ppo_humanoid.yaml for ParSet1 and 100000 training epochs
- ase/data/cfg/train/rlg/common_ppo_humanoid_fullep.yaml for ParSet1 and 233334 training epochs
- ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml for ParSet2 and 100000 training epochs
- ase/data/cfg/train/rlg/common_ppo_humanoid_v3_fullep.yaml for ParSet2 and 233334 training epochs

For `<path-to-the-motion-dataset-file>` you can use any of the yaml files under ase/data/motions. Alternatively, you can put directly motion file (file ending in '.npy') to imitate a single motion. Here we wouls also like to note that the file that have a 'm2' on the name correspond to the model using 5 trackers (for that model we did a separate retargeting and therefore had additional files).

Here is an example for pd-control, r<sub>t,2</sub>, 36Hz, tracker setup H+4C, ParSet2 and 100000 training epochs configuration with the dataset having extra movements for jumping motoins:

```
python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/m2/dataset_plusjumpreduced_train.yaml --headless
```

# Testing commands
### Testing imitation of a motion/various motions
The general form is:
```
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env <path-to-env-config> --cfg_train <path-to-train-config> --motion_file <path-to-the-motion-dataset-file> --checkpoint <path-to-network-checkpoint> --num_envs 10 --headless
```
For values of `<path-to-env-config>`,`<path-to-train-config>` and `<path-to-the-motion-dataset-file>` you have the same options as in the training configurations.

For `<path-to-network-checkpoint>` you use the path to the weights of a trained policy.

You can use another number of environments, not necessary 10.


Here is an example for pd-control, r<sub>t,2</sub>, 36Hz, tracker setup H+2C and ParSet2 configuration tested on the locomotion movements of LAFAN:

```
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafanlocomotion_test.yaml --checkpoint output/HumanoidImitation_20-06-06-19-02/nn/HumanoidImitation.pth --num_envs 10 --headless
```
### Testing real-time control
In order to run the application in real-time we have to first have to instalL [MQTT](https://mqtt.org/) in Ubuntu:
```
sudo apt install -y mosquitto
```
Our implementation in python expects that the MQTT broker is in the localhost and listening at port 1883. If needed this can be modified in the script ase/real_time/mqtt.py. Here we continue showing the steps that we perform with the localhost and port 1883. We first create a file named 'mosquitto.conf' and the content of it is:
```
listener 1883 0.0.0.0
allow_anonymous true
```
Then on the command line we run the follwing commands
```
sudo systemctl stop mosquitto
mosquitto -c mosquitto.conf
```
Now we can run the simulation in python and the Unity application. The command for the simulation is :

```
python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env <path-to-env-config> --cfg_train <path-to-train-config> --motion_file <path-to-the-motion-dataset-file> --checkpoint <path-to-network-checkpoint>  --num_envs 1 --real_time
```
Here you cans use more environments than just 1 if you want. Furthermore, the motions of `<path-to-the-motion-dataset-file>` are not really used. W

# Recording
In order to record the simulation you can run this command on separate command line:
```
python ase/video_processing/window_recorder.py
```
After this, you can start the simulation (with rendering activated). The script 'window_recorder.py' is designed to search for a window called 'Isaac Gym' and record it until it closes. If after some time the simulation has not been started, 'window_recorder.py' will stop searching for a window and terminate.
