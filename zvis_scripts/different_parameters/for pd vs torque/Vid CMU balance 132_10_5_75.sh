cd ../../..
export PYTHONPATH=/home/erick/MotionProjs/ASE_MODS/lib:$PYTHONPATH




#python ase/video_processing/video_stacker_with_duration_change.py --video_files_paths /home/erick/Videos/exp_vids/different_parameters/for_control/bal_132_10_5_75/gt.avi /home/erick/Videos/exp_vids/different_parameters/for_control/bal_132_10_5_75/alt_rt2_to.avi /home/erick/Videos/exp_vids/different_parameters/for_control/bal_132_10_5_75/alt_rt2_pd.avi --sync_times_start_and_end 2.03 4.25 2.03 3.9 2.03 3.51 --orientation vertical


python ase/video_processing/video_stacker.py --video_files_paths /home/erick/Videos/exp_vids/different_parameters/for_control/bal_132_10_5_75/gt.avi /home/erick/Videos/exp_vids/different_parameters/for_control/bal_132_10_5_75/alt_rt2_to.avi /home/erick/Videos/exp_vids/different_parameters/for_control/bal_132_10_5_75/alt_rt2_pd.avi --sync_times_start_and_end 2.03 4.25 2.03 3.9 2.03 3.51 --orientation vertical






