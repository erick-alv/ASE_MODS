cd ..
export LD_LIBRARY_PATH=/home/erick/anaconda3/envs/rlgpu_ase/lib:$LD_LIBRARY_PATH
python ase/output_recorder/window_recorder.py &


#Reward PAR, PD, AltPar, (72Hz)

python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps_1_72/HumanoidImitation_02-07-20-53-34/nn/HumanoidImitation_000100000.pth  --num_envs 1 --real_time --print_camera --init_camera_pat "0.42 5.33 2.87 0.42 0 0"