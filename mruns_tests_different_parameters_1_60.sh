export LD_LIBRARY_PATH=/home/erick/anaconda3/envs/rlgpu_ase/lib:$LD_LIBRARY_PATH

###QuestSim parameters

##rew QuestSim torque
#DONE
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps/HumanoidImitation_23-06-16-34-48/nn/HumanoidImitation.pth --num_envs 10 --headless

#
###rew PenaltyAndReach torque
#DONE
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps/HumanoidImitation_23-06-16-34-48/nn/HumanoidImitation.pth --num_envs 10 --headless

#rew QuestSim torque
#DONE
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps/HumanoidImitation_23-06-16-34-48/nn/HumanoidImitation.pth --num_envs 10 --headless
#
##
#Done
####rew PenaltyAndReach torque
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps/HumanoidImitation_23-06-16-34-48/nn/HumanoidImitation.pth --num_envs 10 --headless





#######v3 parameters
#rew QuestSim torque trained with rewQuestSim
#Done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps/HumanoidImitation_25-06-03-10-31/nn/HumanoidImitation.pth --num_envs 10 --headless


##rew PenaltyAndReach torque trained with rewQuestSim
#Done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps/HumanoidImitation_25-06-03-10-31/nn/HumanoidImitation.pth --num_envs 10 --headless


#rew QuestSim torque trained with rewQuestSim
#Done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps/HumanoidImitation_25-06-03-10-31/nn/HumanoidImitation.pth --num_envs 10 --headless


##rew PenaltyAndReach torque trained with rewQuestSim
#Done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps/HumanoidImitation_25-06-03-10-31/nn/HumanoidImitation.pth --num_envs 10 --headless



#
###rew QuestSim torque trained with rewPenaltyAndReach
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps/HumanoidImitation_27-06-03-28-29/nn/HumanoidImitation.pth --num_envs 10 --headless


##rew PenaltyAndReach torque trained with rewPenaltyAndReach
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps/HumanoidImitation_27-06-03-28-29/nn/HumanoidImitation.pth --num_envs 10 --headless

python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps/HumanoidImitation_27-06-03-28-29/nn/HumanoidImitation.pth --num_envs 10 --headless


##rew PenaltyAndReach torque trained with rewPenaltyAndReach
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps/HumanoidImitation_27-06-03-28-29/nn/HumanoidImitation.pth --num_envs 10 --headless
#
#
#
###rewQuestSim pd trained with rewPenaltyAndReach
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_pd.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint TODO --num_envs 10 --headless
#
###
####rewPenaltyAndReach pd trained with rewPenaltyAndReach
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint TODO --num_envs 10 --headless









