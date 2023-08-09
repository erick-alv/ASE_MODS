cd ../../..
export PYTHONPATH=/home/erick/MotionProjs/ASE_MODS/lib:$PYTHONPATH

python ase/video_processing/video_stacker_with_duration_change.py --video_files_paths /home/erick/Videos/exp_vids/different_parameters/for_hz/dataset-2_walk-turn-left_active_020_diff/gt.avi /home/erick/Videos/exp_vids/different_parameters/for_hz/dataset-2_walk-turn-left_active_020_diff/rt2_to_alt_36.avi /home/erick/Videos/exp_vids/different_parameters/for_hz/dataset-2_walk-turn-left_active_020_diff/rt2_to_alt_72.avi --sync_times_start_and_end 1.96 4.09 2.06 4.54 2.06 7.8 --orientation vertical

python ase/video_processing/video_stacker.py --video_files_paths /home/erick/Videos/exp_vids/different_parameters/for_hz/dataset-2_walk-turn-left_active_020_diff/gt.avi /home/erick/Videos/exp_vids/different_parameters/for_hz/dataset-2_walk-turn-left_active_020_diff/rt2_to_alt_36.avi /home/erick/Videos/exp_vids/different_parameters/for_hz/dataset-2_walk-turn-left_active_020_diff/rt2_to_alt_72.avi --sync_times_start_and_end 1.96 4.09 2.06 4.54 2.06 7.8 --orientation vertical
