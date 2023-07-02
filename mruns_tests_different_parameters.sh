export LD_LIBRARY_PATH=/home/erick/anaconda3/envs/rlgpu_ase/lib:$LD_LIBRARY_PATH

###1/36
###QuestSim parameters



##rew QuestSim torque
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps/HumanoidImitation_11-06-12-25-55/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless
#
#
###rew PenaltyAndReach torque
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps/HumanoidImitation_11-06-12-25-55/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless
#
###rew QuestSim torque
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps/HumanoidImitation_15-06-18-49-50/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless
#
#
###rew PenaltyAndReach torque
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps/HumanoidImitation_15-06-18-49-50/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless

#QuestSim pd
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps/HumanoidImitation_20-06-16-24-51/nn/HumanoidImitation.pth --num_envs 10 --headless





###PenaltyAndReach pd
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps/HumanoidImitation_20-06-16-24-51/nn/HumanoidImitation.pth --num_envs 10 --headless

###!!!!from here on
###v3 parameters
##rew QuestSim torque
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps/HumanoidImitation_19-06-01-53-04/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless


##rew PenaltyAndReach torque
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps/HumanoidImitation_19-06-01-53-04/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless

##rew QuestSim torque
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps/HumanoidImitation_14-06-16-45-17/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless
#
#
##Below this already done
###rew PenaltyAndReach torque
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps/HumanoidImitation_14-06-16-45-17/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless
#


##QuestSim pd
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps/HumanoidImitation_20-06-06-19-02/nn/HumanoidImitation.pth --num_envs 10 --headless
#
##
###PenaltyAndReach pd
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps/HumanoidImitation_20-06-06-19-02/nn/HumanoidImitation.pth --num_envs 10 --headless









