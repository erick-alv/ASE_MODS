cd ../../..
export LD_LIBRARY_PATH=/home/erick/anaconda3/envs/rlgpu_ase/lib:$LD_LIBRARY_PATH

#python ase/output_recorder/window_recorder.py &

##aaaobstacles2_subjaaaaaect2
##this is better run1_subjaaaaaect2

#gt
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/lafan_motions_retargeted/180/run1_subject5.npy --checkpoint output_final_exps/HumanoidImitation_20-06-16-24-51/nn/HumanoidImitation_000100000.pth --num_envs 1 --debug_sync --init_camera_pat "1.0 -7.2 4.8 1.0 -2.5 1.0" --print_camera


#rt_2, pd, QuestSimPar


python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/lafan_motions_retargeted/180/run1_subject5.npy --checkpoint output_final_exps/HumanoidImitation_20-06-16-24-51/nn/HumanoidImitation_000100000.pth --num_envs 1 --init_camera_pat "1.0 -7.2 4.8 1.0 -2.5 1.0"

#vs
#rt_2, pd, AltPar

python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/lafan_motions_retargeted/180/run1_subject5.npy --checkpoint output_final_exps/HumanoidImitation_20-06-06-19-02/nn/HumanoidImitation_000100000.pth --num_envs 1 --init_camera_pat "1.0 -7.2 4.8 1.0 -2.5 1.0"
