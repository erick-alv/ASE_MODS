cd ../..
export PYTHONPATH=/home/erick/MotionProjs/ASE_MODS/lib:$PYTHONPATH

python ase/video_processing/video_stacker_with_duration_change.py --video_files_paths /home/erick/Videos/exp_vids/plus_motions/jump/83_49/gt.avi /home/erick/Videos/exp_vids/plus_motions/jump/83_49/H+2C.avi /home/erick/Videos/exp_vids/plus_motions/jump/83_49/H+4C.avi --sync_times_start_and_end 2.03 4.8 2.12 4.16 2.08 4.38 --orientation vertical

python ase/video_processing/video_stacker.py --video_files_paths /home/erick/Videos/exp_vids/plus_motions/jump/83_49/gt.avi /home/erick/Videos/exp_vids/plus_motions/jump/83_49/H+2C.avi /home/erick/Videos/exp_vids/plus_motions/jump/83_49/H+4C.avi --sync_times_start_and_end 2.03 4.8 2.12 4.16 2.08 4.38 --orientation vertical

