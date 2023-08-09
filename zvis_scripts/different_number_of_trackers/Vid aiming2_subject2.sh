cd ../..
export PYTHONPATH=/home/erick/MotionProjs/ASE_MODS/lib:$PYTHONPATH


python ase/video_processing/video_stacker_with_duration_change.py --video_files_paths /home/erick/Videos/exp_vids/different_number_of_trackers/aiming2_subject2/gt.avi /home/erick/Videos/exp_vids/different_number_of_trackers/aiming2_subject2/par_one.avi /home/erick/Videos/exp_vids/different_number_of_trackers/aiming2_subject2/par_three.avi /home/erick/Videos/exp_vids/different_number_of_trackers/aiming2_subject2/par_five.avi --sync_times_start_and_end 3.41 13.61 3.28 11.35 3.35 11.35 3.38 11.35 --orientation vertical

python ase/video_processing/video_stacker.py --video_files_paths /home/erick/Videos/exp_vids/different_number_of_trackers/aiming2_subject2/gt.avi /home/erick/Videos/exp_vids/different_number_of_trackers/aiming2_subject2/par_one.avi /home/erick/Videos/exp_vids/different_number_of_trackers/aiming2_subject2/par_three.avi /home/erick/Videos/exp_vids/different_number_of_trackers/aiming2_subject2/par_five.avi --sync_times_start_and_end 3.41 13.61 3.28 11.35 3.35 11.35 3.38 11.35 --orientation vertical



