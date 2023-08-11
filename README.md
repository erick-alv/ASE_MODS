# ASE_MODS

Here we provide our reimplementation of the method presented in "QuestSim: Human Motion Tracking from Sparse Sensors with Simulated Avatars" (https://arxiv.org/abs/2209.09391). We base our code on  the repository of the authors of
"ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters" (https://xbpeng.github.io/projects/ASE/index.html). 


# Requirements

# Installation

# Dataset 

# Network weigths

# Training commands

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
Then on the command line we run the follwinf commands
sudo systemctl stop mosquitto
```
sudo systemctl stop mosquitto
mosquitto -c mosquitto.conf
```
Now we can run the simulation in python and the Unity appication. The command for the simulation is :

```
TODO
```
