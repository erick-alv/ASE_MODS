cd ../../..
export LD_LIBRARY_PATH=/home/erick/anaconda3/envs/rlgpu_ase/lib:$LD_LIBRARY_PATH

#python ase/output_recorder/window_recorder.py &
##show pd are very similar walk with curve 45_aaaa01_amp
##bad for 72 f127_08_afewmp

#gt
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_motions_retargeted/180/locomotion/69_61_amp.npy --checkpoint output_final_exps/HumanoidImitation_20-06-06-19-02/nn/HumanoidImitation_000100000.pth --num_envs 1 --debug_sync --init_camera_pat "1.0 -7.2 4.8 1.0 -2.5 1.0" --print_camera


#rt_2, pd, AltPar (36Hz)
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_motions_retargeted/180/locomotion/69_61_amp.npy --checkpoint output_final_exps/HumanoidImitation_20-06-06-19-02/nn/HumanoidImitation_000100000.pth --num_envs 1 --init_camera_pat "1.0 -7.2 4.8 1.0 -2.5 1.0"

#rt_2, pd, AltPar (72Hz)
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_motions_retargeted/180/locomotion/69_61_amp.npy --checkpoint output_final_exps_1_72/HumanoidImitation_02-07-20-53-34/nn/HumanoidImitation_000100000.pth --num_envs 1 --init_camera_pat "1.0 -7.2 4.8 1.0 -2.5 1.0"


#rt_2, torque, AltPar (36Hz)
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_motions_retargeted/180/locomotion/69_61_amp.npy --checkpoint output_final_exps/HumanoidImitation_14-06-16-45-17/nn/HumanoidImitation_000100000.pth --num_envs 1 --init_camera_pat "1.0 -7.2 4.8 1.0 -2.5 1.0"

#rt_2, torque, AltPar (72Hz)
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_motions_retargeted/180/locomotion/69_61_amp.npy --checkpoint output_final_exps_1_72/HumanoidImitation_03-07-06-17-52/nn/HumanoidImitation_000100000.pth --num_envs 1 --init_camera_pat "1.0 -7.2 4.8 1.0 -2.5 1.0"



