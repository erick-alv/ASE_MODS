export LD_LIBRARY_PATH=/home/erick/anaconda3/envs/rlgpu_ase/lib:$LD_LIBRARY_PATH

#example for VM
#nohup python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach_v01.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --headless &

#to continue run
#python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --checkpoint output/HumanoidImitation_22-06-17-27-18/nn/HumanoidImitation_000080000.pth --resume 1 --headless



#######################For different parameters

#####rewQuestSim
######(1/72)
#house
#done
#python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --headless

# done
#python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --headless

######rewPenaltyAndReach
#done house
#python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --headless

#done at VM
#nohup python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --headless &

#####################For different amount of trackers

####One
#Done
#python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/humanoid_imitation_vrhOne_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --headless

####Five
# Done
# python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/m2/dataset_questsimreduced_train.yaml --headless

################(1/72)
####One
#done
#python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrhOne_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --checkpoint output/HumanoidImitation_06-07-08-03-53/nn/HumanoidImitation_000030000.pth --resume 1 --headless

####Five
#repeat this one
#done
#nohup python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/m2/dataset_questsimreduced_train.yaml --headless &


#####################For additional motion types

#running casa
python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_plusdancereduced_train.yaml --headless

python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/humanoid_imitation_vrhm_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_plusjumpreduced_train.yaml --headless



#five
#done
#nohup python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/m2/dataset_plusdancereduced_train.yaml --headless &

#running
nohup python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/m2/dataset_plusjumpreduced_train.yaml --headless &






