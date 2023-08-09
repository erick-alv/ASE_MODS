cd ../..
export PYTHONPATH=/home/erick/MotionProjs/ASE_MODS/lib:$PYTHONPATH


python ase/video_processing/video_stacker_with_duration_change.py --video_files_paths /home/erick/Videos/exp_vids/plus_motions/dance/60_15/gt.avi /home/erick/Videos/exp_vids/plus_motions/dance/60_15/H+2C.avi /home/erick/Videos/exp_vids/plus_motions/dance/60_15/H+4C.avi --sync_times_start_and_end 2.16 14.25 2.19 12.28 2.25 12.19 --orientation vertical

python ase/video_processing/video_stacker.py --video_files_paths /home/erick/Videos/exp_vids/plus_motions/dance/60_15/gt.avi /home/erick/Videos/exp_vids/plus_motions/dance/60_15/H+2C.avi /home/erick/Videos/exp_vids/plus_motions/dance/60_15/H+4C.avi --sync_times_start_and_end 2.16 14.25 2.19 12.28 2.25 12.19 --orientation vertical