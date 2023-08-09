cd ../..
export LD_LIBRARY_PATH=/home/erick/anaconda3/envs/rlgpu_ase/lib:$LD_LIBRARY_PATH

python ase/video_processing/window_recorder.py &


#gt
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/zeggs_motions_retargeted/180/conversation/019_Relaxed_3_mirror_x_1_0.npy --checkpoint output_final_Plusmotions/HumanoidImitation_11-07-03-23-10/nn/HumanoidImitation_000100000.pth --num_envs 1 --debug_sync --init_camera_pat "2.4 0.0 1.4 0.0 0.0 1.0" --print_camera


#jump, 3 t
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/zeggs_motions_retargeted/180/conversation/019_Relaxed_3_mirror_x_1_0.npy --checkpoint output_final_Plusmotions/HumanoidImitation_14-07-16-48-52/nn/HumanoidImitation_000100000.pth --num_envs 1 --init_camera_pat "2.4 0.0 1.4 0.0 0.0 1.0"

#jump, 5 t
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/zeggs_motions_retargeted_m2/180/conversation/019_Relaxed_3_mirror_x_1_0.npy --checkpoint output_final_Plusmotions/HumanoidImitation_09-07-05-04-58/nn/HumanoidImitation_000100000.pth --num_envs 1 --init_camera_pat "2.4 0.0 1.4 0.0 0.0 1.0"




