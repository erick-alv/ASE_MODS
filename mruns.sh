export LD_LIBRARY_PATH=/home/erick/anaconda3/envs/rlgpu_ase/lib:$LD_LIBRARY_PATH

#example for VM
#nohup python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach_v01.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --headless &

#for continue run casa
#python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --checkpoint output/HumanoidImitation_22-06-17-27-18/nn/HumanoidImitation_000080000.pth --resume 1 --headless
## m2 (1/36) (just the 6) (v3) it is working
#python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/humanoid_imitation_vrhm2Five.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/m2/dataset_questsimreduced_train.yaml --headless


## max m2 (that is with 10 obs) (1/60) (v3) ((does)) not work)))
#python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrhm2Five_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/m2/dataset_questsimreduced_train.yaml --headless

######to finish the ones of different parameters
#running this one at house DONE
#python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --headless

python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --headless

######to finish the one of different number of trackers
##(1/36)

python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/humanoid_imitation_vrhOne_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --headless

# running house
# python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/m2/dataset_questsimreduced_train.yaml --headless

##(1/60)

python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrhOne_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --headless

#for VM
#running this one
#nohup python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/m2/dataset_questsimreduced_train.yaml --headless &


#############For additional motion types


python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_plusdancereduced_train.yaml --headless

python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/humanoid_imitation_vrhm_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_plusjumpreduced_train.yaml --headless



#five
nohup python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/m2/dataset_plusdancereduced_train.yaml --headless &

python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/m2/dataset_plusjumpreduced_train.yaml --headless






