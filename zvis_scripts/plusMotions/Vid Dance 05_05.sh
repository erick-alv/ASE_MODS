cd ../..
export PYTHONPATH=/home/erick/MotionProjs/ASE_MODS/lib:$PYTHONPATH


python ase/video_processing/video_stacker_with_duration_change.py --video_files_paths /home/erick/Videos/exp_vids/plus_motions/dance/05_05/gt.avi /home/erick/Videos/exp_vids/plus_motions/dance/05_05/H+2C.avi /home/erick/Videos/exp_vids/plus_motions/dance/05_05/H+4C.avi --sync_times_start_and_end 2.12 7.32 2.03 5.93 2.06 6.16 --orientation vertical

python ase/video_processing/video_stacker.py --video_files_paths /home/erick/Videos/exp_vids/plus_motions/dance/05_05/gt.avi /home/erick/Videos/exp_vids/plus_motions/dance/05_05/H+2C.avi /home/erick/Videos/exp_vids/plus_motions/dance/05_05/H+4C.avi --sync_times_start_and_end 2.12 7.32 2.03 5.93 2.06 6.16 --orientation vertical
