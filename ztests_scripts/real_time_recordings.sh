cd ..
export LD_LIBRARY_PATH=/home/erick/anaconda3/envs/rlgpu_ase/lib:$LD_LIBRARY_PATH
python ase/video_processing/window_recorder.py &

#####PD, rt2, QuestSimPar, (36Hz)
## 100 000
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_20-06-16-24-51/nn/HumanoidImitation_000100000.pth  --num_envs 1 --real_time --print_camera --init_camera_pat "4.0 0.0 2.0 0.0 0.0 1.0"
## full (233 333)
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_23-07-08-39-31/nn/HumanoidImitation.pth  --num_envs 1 --real_time --print_camera --init_camera_pat "4.0 0.0 2.0 0.0 0.0 1.0"

#####PD, rt2, AltPar, (36Hz)
## 100 000
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_20-06-06-19-02/nn/HumanoidImitation_000100000.pth  --num_envs 1 --real_time --print_camera --init_camera_pat "4.0 0.0 2.0 0.0 0.0 1.0"

## full (233 333)
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_21-07-20-51-40/nn/HumanoidImitation.pth  --num_envs 1 --real_time --print_camera --init_camera_pat "4.0 0.0 2.0 0.0 0.0 1.0"

#####PD, rt2, AltPar, (72Hz)

## 100 000
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps_1_72/HumanoidImitation_02-07-20-53-34/nn/HumanoidImitation_000100000.pth  --num_envs 1 --real_time --print_camera --init_camera_pat "4.0 0.0 2.0 0.0 0.0 1.0"
## full (233 333)
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps_1_72/HumanoidImitation_17-07-15-53-22/nn/HumanoidImitation.pth  --num_envs 1 --real_time --print_camera --init_camera_pat "4.0 0.0 2.0 0.0 0.0 1.0"


#worse
##torque, rt2, AltPar
## 100 000
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_14-06-16-45-17/nn/HumanoidImitation_000100000.pth  --num_envs 1 --real_time --print_camera --init_camera_pat "4.0 0.0 2.0 0.0 0.0 1.0"

## full (233 333)
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_18-06-05-01-55/nn/HumanoidImitation.pth  --num_envs 1 --real_time --print_camera --init_camera_pat "4.0 0.0 2.0 0.0 0.0 1.0"



##----------------------------------------- With one single tracker ------------------------------
## torque, rt1, QuestSimPar, one
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrhOne.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_DiffTrackers/HumanoidImitation_24-06-23-54-14/nn/HumanoidImitation_000100000.pth  --num_envs 1 --real_time --print_camera --init_camera_pat "4.0 0.0 2.0 0.0 0.0 1.0"

#PD, rt2, AltPar, (36Hz), one
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrhOne_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_DiffTrackers/HumanoidImitation_01-07-07-25-38/nn/HumanoidImitation_000100000.pth  --num_envs 1 --real_time --print_camera --init_camera_pat "4.0 0.0 2.0 0.0 0.0 1.0"

#PD, rt2, AltPar, (72Hz), one
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrhOne_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_DiffTrackers/HumanoidImitation_07-07-18-01-21/nn/HumanoidImitation_000100000.pth  --num_envs 1 --real_time --print_camera --init_camera_pat "4.0 0.0 2.0 0.0 0.0 1.0"



####---------------- Worst Worst ---------
## torque, rt1, QuestSimPar ((is not following at all; so maybe not))
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_11-06-12-25-55/nn/HumanoidImitation_000100000.pth  --num_envs 1 --real_time --print_camera --init_camera_pat "4.0 0.0 2.0 0.0 0.0 1.0"

## full (233 333)
# python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_11-06-12-25-55/nn/HumanoidImitation.pth  --num_envs 1 --real_time --print_camera --init_camera_pat "4.0 0.0 2.0 0.0 0.0 1.0"