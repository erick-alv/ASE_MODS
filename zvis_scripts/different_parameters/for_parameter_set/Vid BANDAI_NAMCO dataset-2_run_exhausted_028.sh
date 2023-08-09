cd ../../..
export PYTHONPATH=/home/erick/MotionProjs/ASE_MODS/lib:$PYTHONPATH

python ase/video_processing/video_stacker_with_duration_change.py --video_files_paths /home/erick/Videos/exp_vids/different_parameters/for_parameter_set/dataset-2_run_exhausted_028/gt.avi /home/erick/Videos/exp_vids/different_parameters/for_parameter_set/dataset-2_run_exhausted_028/rt2_pd_que.avi /home/erick/Videos/exp_vids/different_parameters/for_parameter_set/dataset-2_run_exhausted_028/rt2_pd_alt.avi --sync_times_start_and_end 2.7 3.83 2.7 3.64 2.7 3.74 --orientation vertical

#python ase/video_processing/video_stacker.py --video_files_paths /home/erick/Videos/exp_vids/different_parameters/for_parameter_set/dataset-2_run_exhausted_028/gt.avi /home/erick/Videos/exp_vids/different_parameters/for_parameter_set/dataset-2_run_exhausted_028/rt2_pd_que.avi /home/erick/Videos/exp_vids/different_parameters/for_parameter_set/dataset-2_run_exhausted_028/rt2_pd_alt.avi --sync_times_start_and_end 2.7 3.83 2.7 3.64 2.7 3.74 --orientation vertical





