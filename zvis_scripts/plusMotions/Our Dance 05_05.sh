cd ../..
export LD_LIBRARY_PATH=/home/erick/anaconda3/envs/rlgpu_ase/lib:$LD_LIBRARY_PATH

#python ase/output_recorder/window_recorder.py &


#gt
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_motions_retargeted/180/dance/05_05_amp.npy --checkpoint output_final_Plusmotions/HumanoidImitation_11-07-03-23-10/nn/HumanoidImitation_000100000.pth --num_envs 1 --debug_sync --init_camera_pat "1.0 -7.2 4.8 1.0 -2.5 1.0" --print_camera


#dance, 3 t
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_motions_retargeted/180/dance/05_05_amp.npy --checkpoint output_final_Plusmotions/HumanoidImitation_11-07-03-23-10/nn/HumanoidImitation_000100000.pth --num_envs 1 --init_camera_pat "1.0 -7.2 4.8 1.0 -2.5 1.0"
#dance, 5 t
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_motions_retargeted_m2/180/dance/05_05.npy --checkpoint output_final_Plusmotions/HumanoidImitation_30-06-10-19-08/nn/HumanoidImitation_000100000.pth --num_envs 1 --init_camera_pat "1.0 -7.2 4.8 1.0 -2.5 1.0"





