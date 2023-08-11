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
Download the motion files from TODO and the yaml files from https://syncandshare.lrz.de/getlink/fiV78B2QVEy1hhxKRFvxPk/datasets_yamls.zip.
Please decompress both zip files in the folder ase/data/motions.

# Network weigths
TODO

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
- ase/data/cfg/train/rlg/common_ppo_humanoid_v3_fullep.yaml for ParSet2 and 100000 training epochs
- ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml for ParSet2 and 233334 training epochs

For `<path-to-env-config>` you can use any of the yaml files under ase/data/motions. Alternatively, you can put directly motion file (file ending in '.npy') to imitate a single motion.

Here is an example for TODO configuration:

```
TODO
```


# Testing commands
...
### real-time control
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
Now we can run the simulation in python and the Unity appication. The command for the simulation is :

```
TODO
```
