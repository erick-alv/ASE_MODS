cd ../../..
export LD_LIBRARY_PATH=/home/erick/anaconda3/envs/rlgpu_ase/lib:$LD_LIBRARY_PATH

#python ase/video_processing/window_recorder.py &

#gt
#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_motions_retargeted/180/balance/132_10_5_75.npy --checkpoint output_final_exps/HumanoidImitation_14-06-16-45-17/nn/HumanoidImitation_000100000.pth --num_envs 1 --debug_sync --init_camera_pat "-0.1 0. 1.0 -1.0 0.0 1.0" --print_camera


#AltPar, rt_2, torque

#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_motions_retargeted/180/balance/132_10_5_75.npy --checkpoint output_final_exps/HumanoidImitation_14-06-16-45-17/nn/HumanoidImitation_000100000.pth --num_envs 1 --init_camera_pat "-0.1 0. 1.0 -1.0 0.0 1.0"

##VS
#AltPar, rt_2, PD

#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_motions_retargeted/180/balance/132_10_5_75.npy --checkpoint output_final_exps/HumanoidImitation_20-06-06-19-02/nn/HumanoidImitation_000100000.pth --num_envs 1 --init_camera_pat "-0.1 0. 1.0 -1.0 0.0 1.0"



