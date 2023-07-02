export LD_LIBRARY_PATH=/home/erick/anaconda3/envs/rlgpu_ase/lib:$LD_LIBRARY_PATH

#run up to 12 of shorts ##Still needs to run with lafan
#1
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps/HumanoidImitation_11-06-12-25-55/nn/HumanoidImitation.pth --num_envs 10 --headless
#
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps/HumanoidImitation_11-06-12-25-55/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless

#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps/HumanoidImitation_11-06-12-25-55/nn/HumanoidImitation_000150000.pth --num_envs 10 --headless
#
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps/HumanoidImitation_11-06-12-25-55/nn/HumanoidImitation_000050000.pth --num_envs 10 --headless

#LAFAN
##still need to run this one
##python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps/HumanoidImitation_11-06-12-25-55/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless


#2
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps/HumanoidImitation_15-06-18-49-50/nn/HumanoidImitation.pth --num_envs 10 --headless
#
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps/HumanoidImitation_15-06-18-49-50/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless

#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps/HumanoidImitation_15-06-18-49-50/nn/HumanoidImitation_000150000.pth --num_envs 10 --headless
#
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps/HumanoidImitation_15-06-18-49-50/nn/HumanoidImitation_000050000.pth --num_envs 10 --headless


#3 still needs to run
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps/HumanoidImitation_19-06-01-53-04/nn/HumanoidImitation.pth --num_envs 10 --headless

#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps/HumanoidImitation_19-06-01-53-04/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless

#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps/HumanoidImitation_19-06-01-53-04/nn/HumanoidImitation_000150000.pth --num_envs 10 --headless

#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps/HumanoidImitation_19-06-01-53-04/nn/HumanoidImitation_000050000.pth --num_envs 10 --headless

#4
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps/HumanoidImitation_18-06-05-01-55/nn/HumanoidImitation.pth --num_envs 10 --headless
#
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps/HumanoidImitation_14-06-16-45-17/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless
#
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps/HumanoidImitation_14-06-16-45-17/nn/HumanoidImitation_000150000.pth --num_envs 10 --headless
#
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps/HumanoidImitation_14-06-16-45-17/nn/HumanoidImitation_000050000.pth --num_envs 10 --headless










