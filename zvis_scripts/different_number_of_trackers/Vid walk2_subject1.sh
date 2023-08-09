cd ../..
export PYTHONPATH=/home/erick/MotionProjs/ASE_MODS/lib:$PYTHONPATH


python ase/video_processing/video_stacker_with_duration_change.py --video_files_paths /home/erick/Videos/exp_vids/different_number_of_trackers/walk2_subject1/gt.avi /home/erick/Videos/exp_vids/different_number_of_trackers/walk2_subject1/72par_one.avi /home/erick/Videos/exp_vids/different_number_of_trackers/walk2_subject1/72par_three.avi /home/erick/Videos/exp_vids/different_number_of_trackers/walk2_subject1/72par_five.avi --sync_times_start_and_end 3.03 15.74 3.03 22.51 3.03 20.32 3.09 18.28 --orientation vertical

python ase/video_processing/video_stacker.py --video_files_paths /home/erick/Videos/exp_vids/different_number_of_trackers/walk2_subject1/gt.avi /home/erick/Videos/exp_vids/different_number_of_trackers/walk2_subject1/72par_one.avi /home/erick/Videos/exp_vids/different_number_of_trackers/walk2_subject1/72par_three.avi /home/erick/Videos/exp_vids/different_number_of_trackers/walk2_subject1/72par_five.avi --sync_times_start_and_end 3.03 15.74 3.03 22.51 3.03 20.32 3.09 18.28 --orientation vertical
