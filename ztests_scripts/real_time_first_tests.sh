cd ..
export LD_LIBRARY_PATH=/home/erick/anaconda3/envs/rlgpu_ase/lib:$LD_LIBRARY_PATH
#python ase/output_recorder/window_recorder.py &


############################1
#Reward QuestSim, torque, QuestSimPar
# vs
#Reward QuestSim, torque, AltPar
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_11-06-12-25-55/nn/HumanoidImitation_000100000.pth --num_envs 1 --real_time
#
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_19-06-01-53-04/nn/HumanoidImitation_000010000.pth --num_envs 1 --real_time

#Which one is better ?
### Reward QuestSim, torque, QuestSimPar

##Reward PAR, torque, QuestSimPar
## vs
##Reward PAR, torque, AltPar
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_15-06-18-49-50/nn/HumanoidImitation_000100000.pth --num_envs 1 --real_time
#
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_14-06-16-45-17/nn/HumanoidImitation_000100000.pth --num_envs 1 --real_time


#which one is better??
#Reward PAR, torque, AltPar

#
echo 2
#############################2
#
###50 000
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_11-06-12-25-55/nn/HumanoidImitation_000050000.pth  --num_envs 1 --real_time
#
#
###100 000
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_11-06-12-25-55/nn/HumanoidImitation_000100000.pth  --num_envs 1 --real_time
#
###150 000
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_11-06-12-25-55/nn/HumanoidImitation_000150000.pth  --num_envs 1 --real_time
#
###233 333
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_11-06-12-25-55/nn/HumanoidImitation.pth  --num_envs 1 --real_time
#
###best
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_11-06-12-25-55/nn/HumanoidImitation_best.pth  --num_envs 1 --real_time
#
#
##the one performing best for winner 1 is
#233 333
#
###---------------------------
#
echo 25
###50 000
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_14-06-16-45-17/nn/HumanoidImitation_000050000.pth  --num_envs 1 --real_time
#
#
###100 000
#
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_14-06-16-45-17/nn/HumanoidImitation_000100000.pth  --num_envs 1 --real_time
#
###150 000
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_14-06-16-45-17/nn/HumanoidImitation_000150000.pth  --num_envs 1 --real_time
#
###233 333
##
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_18-06-05-01-55/nn/HumanoidImitation.pth  --num_envs 1 --real_time
#
###best
##
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_18-06-05-01-55/nn/HumanoidImitation_best.pth  --num_envs 1 --real_time

#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_14-06-16-45-17/nn/HumanoidImitation_best.pth  --num_envs 1 --real_time
#
#233 333; best is not better
#
#
##the one performing best for winner 2 is
#
echo 3
############################3
#
### vs vs vs
##Reward QuestSim, torque, QuestSimPar
## vs
##Reward PAR, torque, AltPar
## vs
##Reward PAR, PD, QuestSimPar
## vs
##Reward PAR, PD, AltPar
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_11-06-12-25-55/nn/HumanoidImitation_000100000.pth --num_envs 1 --real_time
#
# better than before
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_14-06-16-45-17/nn/HumanoidImitation_000100000.pth --num_envs 1 --real_time
#
#even better than before
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_20-06-16-24-51/nn/HumanoidImitation_000100000.pth  --num_envs 1 --real_time


#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_20-06-06-19-02/nn/HumanoidImitation_000100000.pth  --num_envs 1 --real_time
#
#
#
### the winner is
#
#
### Is PD better than torque???
# Yes
#Reward PAR, PD, QuestSimPar
# Reward PAR, PD, AltPar (better if not falls)
echo 4
############################4
#
### vs vs vs
### vs vs vs
##Reward PAR, torque, AltPar, (36Hz)
## vs
##Reward PAR, torque, AltPar, (72Hz)
## vs
##Reward PAR, PD, AltPar, (36Hz)
## vs
##Reward PAR, PD, AltPar, (72Hz)
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_14-06-16-45-17/nn/HumanoidImitation_000100000.pth  --num_envs 1 --real_time
#
#
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps_1_72/HumanoidImitation_03-07-06-17-52/nn/HumanoidImitation_000100000.pth  --num_envs 1 --real_time
#
#
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_20-06-06-19-02/nn/HumanoidImitation_000100000.pth  --num_envs 1 --real_time
#
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps_1_72/HumanoidImitation_02-07-20-53-34/nn/HumanoidImitation_000100000.pth  --num_envs 1 --real_time


##over torque
#both are bad
##over PD
# 72 Hz
## the overall winner is??
#Reward PAR, PD, AltPar, (72Hz)

## is 36Hz or 72Hz performing better???
##for the one working 72 Hz



########################5
echo 5
##winner 3 vs BEST
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_20-06-16-24-51/nn/HumanoidImitation_000050000.pth  --num_envs 1 --real_time

#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_20-06-06-19-02/nn/HumanoidImitation_000050000.pth  --num_envs 1 --real_time


#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_20-06-16-24-51/nn/HumanoidImitation_000100000.pth  --num_envs 1 --real_time
#
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_20-06-06-19-02/nn/HumanoidImitation_000100000.pth  --num_envs 1 --real_time
#
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_20-06-16-24-51/nn/HumanoidImitation_best.pth  --num_envs 1 --real_time
#
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps/HumanoidImitation_20-06-06-19-02/nn/HumanoidImitation_best.pth  --num_envs 1 --real_time
##Winners
##Reward PAR, PD, QuestSimPar 50000
## vs
##Reward PAR, PD, AltPar (2 times) best and the longest


echo 55
#
###winner 4 vs BEST
#Reward PAR, PD, AltPar, (72Hz)

#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps_1_72/HumanoidImitation_02-07-20-53-34/nn/HumanoidImitation_000050000.pth  --num_envs 1 --real_time
#
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps_1_72/HumanoidImitation_02-07-20-53-34/nn/HumanoidImitation_000100000.pth  --num_envs 1 --real_time
#
#python ase/run.py --test --task HumanoidImitationTrack --algo_name common_real_time --cfg_env ase/data/cfg/other_setup/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/cmu_temp_retargeted/07_01_cmu_amp.npy --checkpoint output_final_exps_1_72/HumanoidImitation_02-07-20-53-34/nn/HumanoidImitation_best.pth  --num_envs 1 --real_time

#not the best; but the longest



#########6 one tracker vs 3 trackers

