cd ../..
export PYTHONPATH=/home/erick/MotionProjs/ASE_MODS/lib:$PYTHONPATH


python ase/video_processing/video_stacker_with_duration_change.py --video_files_paths /home/erick/Videos/exp_vids/plus_motions/dance/dance2_subject2/gt.avi /home/erick/Videos/exp_vids/plus_motions/dance/dance2_subject2/H+2C.avi /home/erick/Videos/exp_vids/plus_motions/dance/dance2_subject2/H+4C.avi --sync_times_start_and_end 2.96 43.03 2.99 32.22 2.86 32.7 --orientation vertical



python ase/video_processing/video_stacker.py --video_files_paths /home/erick/Videos/exp_vids/plus_motions/dance/dance2_subject2/gt.avi /home/erick/Videos/exp_vids/plus_motions/dance/dance2_subject2/H+2C.avi /home/erick/Videos/exp_vids/plus_motions/dance/dance2_subject2/H+4C.avi --sync_times_start_and_end 2.93 43.03 2.96 32.22 2.83 32.7 --orientation vertical
