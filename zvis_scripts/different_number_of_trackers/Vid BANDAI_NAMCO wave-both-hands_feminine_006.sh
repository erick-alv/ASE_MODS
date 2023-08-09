cd ../..
export PYTHONPATH=/home/erick/MotionProjs/ASE_MODS/lib:$PYTHONPATH

python ase/video_processing/video_stacker_with_duration_change.py --video_files_paths /home/erick/Videos/exp_vids/different_number_of_trackers/dataset-2_wave-both-hands_feminine_006/gt.avi /home/erick/Videos/exp_vids/different_number_of_trackers/dataset-2_wave-both-hands_feminine_006/72par_one.avi /home/erick/Videos/exp_vids/different_number_of_trackers/dataset-2_wave-both-hands_feminine_006/72par_three.avi /home/erick/Videos/exp_vids/different_number_of_trackers/dataset-2_wave-both-hands_feminine_006/72par_five.avi --sync_times_start_and_end 2.0 8.06 2.06 11.16 2.0 11.16 2.06 11.16 --orientation vertical

python ase/video_processing/video_stacker.py --video_files_paths /home/erick/Videos/exp_vids/different_number_of_trackers/dataset-2_wave-both-hands_feminine_006/gt.avi /home/erick/Videos/exp_vids/different_number_of_trackers/dataset-2_wave-both-hands_feminine_006/72par_one.avi /home/erick/Videos/exp_vids/different_number_of_trackers/dataset-2_wave-both-hands_feminine_006/72par_three.avi /home/erick/Videos/exp_vids/different_number_of_trackers/dataset-2_wave-both-hands_feminine_006/72par_five.avi --sync_times_start_and_end 2.0 8.06 2.06 11.16 2.0 11.16 2.06 11.16 --orientation vertical




