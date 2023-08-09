cd ../../..
export PYTHONPATH=/home/erick/MotionProjs/ASE_MODS/lib:$PYTHONPATH



python ase/video_processing/video_stacker_with_duration_change.py --video_files_paths /home/erick/Videos/exp_vids/different_parameters/for_parameter_set/run1_subject2/gt.avi /home/erick/Videos/exp_vids/different_parameters/for_parameter_set/run1_subject2/rt2_pd_que.avi /home/erick/Videos/exp_vids/different_parameters/for_parameter_set/run1_subject2/rt2_pd_alt.avi --sync_times_start_and_end 3.61 17.28 3.61 17.35 3.61 16.99 --orientation vertical

python ase/video_processing/video_stacker.py --video_files_paths /home/erick/Videos/exp_vids/different_parameters/for_parameter_set/run1_subject2/gt.avi /home/erick/Videos/exp_vids/different_parameters/for_parameter_set/run1_subject2/rt2_pd_que.avi /home/erick/Videos/exp_vids/different_parameters/for_parameter_set/run1_subject2/rt2_pd_alt.avi --sync_times_start_and_end 3.61 17.28 3.61 17.35 3.61 16.99 --orientation vertical




