export LD_LIBRARY_PATH=/home/erick/anaconda3/envs/rlgpu_ase/lib:$LD_LIBRARY_PATH

python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/small_humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/small_common_ppo_humanoid.yaml --motion_file ase/data/motions/cmu_motions_retargeted/180/locomotion/07_01_amp.npy --headless

python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/small_humanoid_imitation_vrh_rewPenaltyAndReach_prerew.yaml --cfg_train ase/data/cfg/train/rlg/small_common_ppo_humanoid.yaml --motion_file ase/data/motions/cmu_motions_retargeted/180/locomotion/07_01_amp.npy --headless

