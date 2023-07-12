export LD_LIBRARY_PATH=/home/erick/anaconda3/envs/rlgpu_ase/lib:$LD_LIBRARY_PATH

###Using reachAndPenaltyReward
###PD control
###V4 of paramaters


#####################For different amount of trackers

####One
#Done
python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/humanoid_imitation_vrhOne_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v4_fullep.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --headless

####Three
python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v4_fullep.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --headless


####Five
# Done
nohup python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v4_fullep.yaml --motion_file ase/data/motions/m2/dataset_questsimreduced_train.yaml --headless &









