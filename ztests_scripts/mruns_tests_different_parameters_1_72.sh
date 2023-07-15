cd ..
export LD_LIBRARY_PATH=/home/erick/anaconda3/envs/rlgpu_ase/lib:$LD_LIBRARY_PATH

###################Trained with QuestSimReward
#######Normal parameters
#####torque
#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps_1_72/HumanoidImitation_09-07-08-30-33/nn/HumanoidImitation.pth --num_envs 10 --headless
#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps_1_72/HumanoidImitation_09-07-08-30-33/nn/HumanoidImitation.pth --num_envs 10 --headless

#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps_1_72/HumanoidImitation_09-07-08-30-33/nn/HumanoidImitation.pth --num_envs 10 --headless

#Still TODO run
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps_1_72/HumanoidImitation_09-07-08-30-33/nn/HumanoidImitation.pth --num_envs 10 --headless
#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_lafanlocomotion_test.yaml --checkpoint output_final_exps_1_72/HumanoidImitation_09-07-08-30-33/nn/HumanoidImitation.pth --num_envs 10 --headless

#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_lafanlocomotion_test.yaml --checkpoint output_final_exps_1_72/HumanoidImitation_09-07-08-30-33/nn/HumanoidImitation.pth --num_envs 10 --headless

########V3 parameter
######torque
#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps_1_72/HumanoidImitation_07-07-18-24-42/nn/HumanoidImitation.pth --num_envs 10 --headless

#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps_1_72/HumanoidImitation_07-07-18-24-42/nn/HumanoidImitation.pth --num_envs 10 --headless
##STILL TODO run
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps_1_72/HumanoidImitation_07-07-18-24-42/nn/HumanoidImitation.pth --num_envs 10 --headless
##STILL TODO run
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps_1_72/HumanoidImitation_07-07-18-24-42/nn/HumanoidImitation.pth --num_envs 10 --headless

#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafanlocomotion_test.yaml --checkpoint output_final_exps_1_72/HumanoidImitation_07-07-18-24-42/nn/HumanoidImitation.pth --num_envs 10 --headless

#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafanlocomotion_test.yaml --checkpoint output_final_exps_1_72/HumanoidImitation_07-07-18-24-42/nn/HumanoidImitation.pth --num_envs 10 --headless



###################Trained with penaltyAndReach
########Normal parameters
#none

########V3 parameter
######torque
#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps_1_72/HumanoidImitation_03-07-06-17-52/nn/HumanoidImitation.pth --num_envs 10 --headless

#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps_1_72/HumanoidImitation_03-07-06-17-52/nn/HumanoidImitation.pth --num_envs 10 --headless

##STILL TODO run
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps_1_72/HumanoidImitation_03-07-06-17-52/nn/HumanoidImitation.pth --num_envs 10 --headless
##STILL TODO
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps_1_72/HumanoidImitation_03-07-06-17-52/nn/HumanoidImitation.pth --num_envs 10 --headless

#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafanlocomotion_test.yaml --checkpoint output_final_exps_1_72/HumanoidImitation_03-07-06-17-52/nn/HumanoidImitation.pth --num_envs 10 --headless
#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafanlocomotion_test.yaml --checkpoint output_final_exps_1_72/HumanoidImitation_03-07-06-17-52/nn/HumanoidImitation.pth --num_envs 10 --headless


######pd
#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_pd.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps_1_72/HumanoidImitation_02-07-20-53-34/nn/HumanoidImitation.pth --num_envs 10 --headless

#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_exps_1_72/HumanoidImitation_02-07-20-53-34/nn/HumanoidImitation.pth --num_envs 10 --headless

#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_pd.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps_1_72/HumanoidImitation_02-07-20-53-34/nn/HumanoidImitation.pth --num_envs 10 --headless

#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafan_test.yaml --checkpoint output_final_exps_1_72/HumanoidImitation_02-07-20-53-34/nn/HumanoidImitation.pth --num_envs 10 --headless

##STILL TODO
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_pd.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafanlocomotion_test.yaml --checkpoint output_final_exps_1_72/HumanoidImitation_02-07-20-53-34/nn/HumanoidImitation.pth --num_envs 10 --headless










