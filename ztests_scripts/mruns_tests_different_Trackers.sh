cd ..
export LD_LIBRARY_PATH=/home/erick/anaconda3/envs/rlgpu_ase/lib:$LD_LIBRARY_PATH

###just PenaltyAndReach
######The one with QuestSim Setup

##one

#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrhOne_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_lafansimilar_test.yaml --checkpoint output_final_DiffTrackers/HumanoidImitation_24-06-23-54-14/nn/HumanoidImitation.pth --num_envs 10 --headless


##three

#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_lafansimilar_test.yaml --checkpoint output_final_exps/HumanoidImitation_11-06-12-25-55/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless


##five

#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrhm2Five_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/m2/dataset_lafansimilar_test.yaml --checkpoint output_final_DiffTrackers/HumanoidImitation_25-06-12-19-56/nn/HumanoidImitation.pth --num_envs 10 --headless



######The one with best setup (v3, pd)

##one

#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrhOne_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_DiffTrackers/HumanoidImitation_01-07-07-25-38/nn/HumanoidImitation.pth --num_envs 10 --headless
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrhOne_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_DiffTrackers/HumanoidImitation_01-07-07-25-38/nn/HumanoidImitation.pth --num_envs 10 --headless
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrhOne_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafansimilar_test.yaml --checkpoint output_final_DiffTrackers/HumanoidImitation_01-07-07-25-38/nn/HumanoidImitation.pth --num_envs 10 --headless
#

##three
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafansimilar_test.yaml --checkpoint output_final_exps/HumanoidImitation_20-06-06-19-02/nn/HumanoidImitation.pth --num_envs 10 --headless

###five
#
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/m2/dataset_questsim_test.yaml --checkpoint output_final_DiffTrackers/HumanoidImitation_29-06-03-38-01/nn/HumanoidImitation.pth --num_envs 10 --headless
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/m2/dataset_lafan_test.yaml --checkpoint output_final_DiffTrackers/HumanoidImitation_29-06-03-38-01/nn/HumanoidImitation.pth --num_envs 10 --headless
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/m2/dataset_lafansimilar_test.yaml --checkpoint output_final_DiffTrackers/HumanoidImitation_29-06-03-38-01/nn/HumanoidImitation.pth --num_envs 10 --headless
#
#try until here
####################################(1/72)
#
###one
#
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrhOne_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_DiffTrackers/HumanoidImitation_07-07-18-01-21/nn/HumanoidImitation.pth --num_envs 10 --headless
#
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrhOne_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_DiffTrackers/HumanoidImitation_07-07-18-01-21/nn/HumanoidImitation.pth --num_envs 10 --headless
#
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrhOne_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafansimilar_test.yaml --checkpoint output_final_DiffTrackers/HumanoidImitation_07-07-18-01-21/nn/HumanoidImitation.pth --num_envs 10 --headless
#
#
#
#
##three
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafansimilar_test.yaml --checkpoint output_final_exps_1_72/HumanoidImitation_02-07-20-53-34/nn/HumanoidImitation.pth --num_envs 10 --headless

#TODO the other 2 ones with three are missing!!!

###five
#
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/m2/dataset_questsim_test.yaml --checkpoint output_final_DiffTrackers/HumanoidImitation_05-07-20-07-52/nn/HumanoidImitation.pth --num_envs 10 --headless
#
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/m2/dataset_lafan_test.yaml --checkpoint output_final_DiffTrackers/HumanoidImitation_05-07-20-07-52/nn/HumanoidImitation.pth --num_envs 10 --headless
#
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/m2/dataset_lafansimilar_test.yaml --checkpoint output_final_DiffTrackers/HumanoidImitation_05-07-20-07-52/nn/HumanoidImitation.pth --num_envs 10 --headless






