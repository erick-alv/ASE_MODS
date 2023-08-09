cd ../..
export PYTHONPATH=/home/erick/MotionProjs/ASE_MODS/lib:$PYTHONPATH


python ase/video_processing/video_stacker_with_duration_change.py --video_files_paths /home/erick/Videos/exp_vids/different_number_of_trackers/walk3_subject2/gt.avi /home/erick/Videos/exp_vids/different_number_of_trackers/walk3_subject2/questsim_one.avi /home/erick/Videos/exp_vids/different_number_of_trackers/walk3_subject2/questsim_three.avi /home/erick/Videos/exp_vids/different_number_of_trackers/walk3_subject2/questsim_five.avi --sync_times_start_and_end 3.06 12.45 3.03 9.7 3.06 9.16 3.06 9.16 --orientation vertical

python ase/video_processing/video_stacker.py --video_files_paths /home/erick/Videos/exp_vids/different_number_of_trackers/walk3_subject2/gt.avi /home/erick/Videos/exp_vids/different_number_of_trackers/walk3_subject2/questsim_one.avi /home/erick/Videos/exp_vids/different_number_of_trackers/walk3_subject2/questsim_three.avi /home/erick/Videos/exp_vids/different_number_of_trackers/walk3_subject2/questsim_five.avi --sync_times_start_and_end 3.06 12.45 3.03 9.7 3.06 9.16 3.06 9.16 --orientation vertical
