export LD_LIBRARY_PATH=/home/erick/anaconda3/envs/rlgpu_ase/lib:$LD_LIBRARY_PATH

python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --headless

#for test 2
nohup python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach_v02.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --headless &


#vm1 erick
#now running here instead
nohup python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --headless &

#vm2 papa
nohup python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --headless &

#vm3 isaac
#runned test
nohup python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach_v01.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --headless &
#for final exp (stopped)
nohup python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --headless &



#for run casa
python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3_fullep.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --checkpoint output/HumanoidImitation_14-06-16-45-17/nn/HumanoidImitation_000180000.pth --resume 1 --headless

#new one
python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3_fullep.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --headless



# run uni
python ase/run.py --task HumanoidImitationTrack --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_fullep.yaml --motion_file ase/data/motions/dataset_questsimreduced_train.yaml --headless






