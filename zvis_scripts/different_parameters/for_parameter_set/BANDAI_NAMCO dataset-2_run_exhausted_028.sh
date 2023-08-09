cd ../../..
export LD_LIBRARY_PATH=/home/erick/anaconda3/envs/rlgpu_ase/lib:$LD_LIBRARY_PATH

python ase/video_processing/window_recorder.py &


#gt
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/bandai_namco_motions_retargeted/180/locomotion/dataset-2_run_exhausted_028.npy --checkpoint output_final_exps/HumanoidImitation_20-06-16-24-51/nn/HumanoidImitation_000100000.pth --num_envs 1 --debug_sync --init_camera_pat "-1.5 -3.8 2.11 -1.5 0.0 1.0" --print_camera


#rt_2, pd, QuestSimPar


#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/bandai_namco_motions_retargeted/180/locomotion/dataset-2_run_exhausted_028.npy --checkpoint output_final_exps/HumanoidImitation_20-06-16-24-51/nn/HumanoidImitation_000100000.pth --num_envs 1 --init_camera_pat "-1.5 -3.8 2.11 -1.5 0.0 1.0"

#vs
#rt_2, pd, AltPar

python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/bandai_namco_motions_retargeted/180/locomotion/dataset-2_run_exhausted_028.npy --checkpoint output_final_exps/HumanoidImitation_20-06-06-19-02/nn/HumanoidImitation_000100000.pth --num_envs 1 --init_camera_pat "-1.5 -3.8 2.11 -1.5 0.0 1.0"
