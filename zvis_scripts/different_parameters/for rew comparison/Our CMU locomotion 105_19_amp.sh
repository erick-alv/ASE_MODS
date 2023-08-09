cd ../../..
export LD_LIBRARY_PATH=/home/erick/anaconda3/envs/rlgpu_ase/lib:$LD_LIBRARY_PATH

python ase/video_processing/window_recorder.py &


#gt
#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_motions_retargeted/180/locomotion/105_19_amp.npy --checkpoint output_final_exps/HumanoidImitation_19-06-01-53-04/nn/HumanoidImitation_000100000.pth --num_envs 1 --debug_sync --init_camera_pat "0.2 3.84 2.8 0.2 -1.0 0.0" --print_camera

#(AltPar, torque) rt_1
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_motions_retargeted/180/locomotion/105_19_amp.npy --checkpoint output_final_exps/HumanoidImitation_19-06-01-53-04/nn/HumanoidImitation_000100000.pth --num_envs 1 --init_camera_pat "0.2 3.84 2.8 0.2 -1.0 0.0"

##VS
#(AltPar, torque) rt_2
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_motions_retargeted/180/locomotion/105_19_amp.npy --checkpoint output_final_exps/HumanoidImitation_14-06-16-45-17/nn/HumanoidImitation_000100000.pth --num_envs 1 --init_camera_pat "0.2 3.84 2.8 0.2 -1.0 0.0"



