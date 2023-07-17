cd ..
export LD_LIBRARY_PATH=/home/erick/anaconda3/envs/rlgpu_ase/lib:$LD_LIBRARY_PATH

##on our BasicES
##RespectiveES
##LAFAN(respective)

########Dance
###Three

#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_Plusmotions/HumanoidImitation_11-07-03-23-10/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless

#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_dance_test.yaml --checkpoint output_final_Plusmotions/HumanoidImitation_11-07-03-23-10/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless

#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafanjustdance_test.yaml --checkpoint output_final_Plusmotions/HumanoidImitation_11-07-03-23-10/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless


###Five
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_Plusmotions/HumanoidImitation_30-06-10-19-08/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless

python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/m2/dataset_dance_test.yaml --checkpoint output_final_Plusmotions/HumanoidImitation_30-06-10-19-08/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless

python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/m2/dataset_lafanjustdance_test.yaml --checkpoint output_final_Plusmotions/HumanoidImitation_30-06-10-19-08/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless

########Jump
###Three

#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_questsim_test.yaml --checkpoint output_final_Plusmotions/HumanoidImitation_14-07-16-48-52/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless
#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_jump_test.yaml --checkpoint output_final_Plusmotions/HumanoidImitation_14-07-16-48-52/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless
#done
#python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrh_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/dataset_lafanjustjump_test.yaml --checkpoint output_final_Plusmotions/HumanoidImitation_14-07-16-48-52/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless


###Five

python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/m2/dataset_questsim_test.yaml --checkpoint output_final_Plusmotions/HumanoidImitation_09-07-05-04-58/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/m2/dataset_jump_test.yaml --checkpoint output_final_Plusmotions/HumanoidImitation_09-07-05-04-58/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless
python ase/run.py --test --algo_name common_test --task HumanoidImitationTrackTest --cfg_env ase/data/cfg/humanoid_imitation_vrhm2Five_pd_rewPenaltyAndReach.yaml --cfg_train ase/data/cfg/train/rlg/common_ppo_humanoid_v3.yaml --motion_file ase/data/motions/m2/dataset_lafanjustjump_test.yaml --checkpoint output_final_Plusmotions/HumanoidImitation_09-07-05-04-58/nn/HumanoidImitation_000100000.pth --num_envs 10 --headless