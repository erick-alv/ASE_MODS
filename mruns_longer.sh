export LD_LIBRARY_PATH=/home/erick/anaconda3/envs/rlgpu_ase/lib:$LD_LIBRARY_PATH


#just 2; #max3


#!!!!Reward PAR, PD, AltPar, (72Hz)
python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3_fullep.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --headless

##Reward PAR, PD, QuestSimPar 50000 or Reward PAR, PD, AltPar (2 times) best and the longest
# Perhaps QuestSimPar to have one a bit longer
#todo


echo 55
#
###winner 4 vs BEST



