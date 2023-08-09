cd ../..
export PYTHONPATH=/home/erick/MotionProjs/ASE_MODS/lib:$PYTHONPATH

python ase/video_processing/video_stacker_with_duration_change.py --video_files_paths /home/erick/Videos/exp_vids/plus_motions/jump/jumps1_subject1/gt.avi /home/erick/Videos/exp_vids/plus_motions/jump/jumps1_subject1/H+2C.avi /home/erick/Videos/exp_vids/plus_motions/jump/jumps1_subject1/H+4C.avi --sync_times_start_and_end 3.16 18.16 3.03 14.38 3.22 14.38 --orientation vertical

python ase/video_processing/video_stacker.py --video_files_paths /home/erick/Videos/exp_vids/plus_motions/jump/jumps1_subject1/gt.avi /home/erick/Videos/exp_vids/plus_motions/jump/jumps1_subject1/H+2C.avi /home/erick/Videos/exp_vids/plus_motions/jump/jumps1_subject1/H+4C.avi --sync_times_start_and_end 3.16 18.16 3.03 14.38 3.22 14.38 --orientation vertical