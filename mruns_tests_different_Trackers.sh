export LD_LIBRARY_PATH=/home/erick/anaconda3/envs/rlgpu_ase/lib:$LD_LIBRARY_PATH

######The one with QuestSim Setup

###just PenaltyAndReach

##one

#test set
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrhOne_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_DiffTrackers/HumanoidImitation_24-06-23-54-14/nn/HumanoidImitation.pth --num_envs 10 --headless

#lafan
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrhOne_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_DiffTrackers/HumanoidImitation_24-06-23-54-14/nn/HumanoidImitation.pth --num_envs 10 --headless


##five

#test set
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrhm2Five_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/m2/dataset_questsim_test.yaml --checkpoint output_final_DiffTrackers/HumanoidImitation_25-06-12-19-56/nn/HumanoidImitation.pth --num_envs 10 --headless
#
##lafan
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrhm2Five_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/m2/dataset_lafan_test.yaml --checkpoint output_final_DiffTrackers/HumanoidImitation_25-06-12-19-56/nn/HumanoidImitation.pth --num_envs 10 --headless


######The one with best setup

###just PenaltyAndReach

##one

#test set

#lafan


##five

##1/60

##one

#test set

#lafan


##five
#test set
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/m2/dataset_questsim_test.yaml --checkpoint output_final_DiffTrackers/HumanoidImitation_27-06-07-33-14/nn/HumanoidImitation.pth --num_envs 10 --headless

#lafan
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/m2/dataset_lafan_test.yaml --checkpoint output_final_DiffTrackers/HumanoidImitation_27-06-07-33-14/nn/HumanoidImitation.pth --num_envs 10 --headless








