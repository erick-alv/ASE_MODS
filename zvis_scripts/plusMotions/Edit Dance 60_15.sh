cd ../..
export PYTHONPATH=/home/erick/MotionProjs/ASE_MODS/lib:$PYTHONPATH


#python ase/video_processing/stack_frame.py --video_files_paths /home/erick/Videos/exp_vids/plus_motions/dance/60_15/gt.avi /home/erick/Videos/exp_vids/plus_motions/dance/60_15/H+2C.avi /home/erick/Videos/exp_vids/plus_motions/dance/60_15/H+4C.avi --sync_times 5.83 5.19 4.93 --orientation vertical --basename dance_60_15

#python ase/video_processing/stack_frame.py --video_files_paths /home/erick/Videos/exp_vids/plus_motions/dance/60_15/gt.avi /home/erick/Videos/exp_vids/plus_motions/dance/60_15/H+2C.avi /home/erick/Videos/exp_vids/plus_motions/dance/60_15/H+4C.avi --sync_times 6.38 7.06 6.86 --orientation vertical --basename dance_60_15

python ase/video_processing/stack_frame.py --video_files_paths /home/erick/Videos/exp_vids/plus_motions/dance/60_15/gt.avi /home/erick/Videos/exp_vids/plus_motions/dance/60_15/H+2C.avi /home/erick/Videos/exp_vids/plus_motions/dance/60_15/H+4C.avi --sync_times 10.22 8.51 7.83 --orientation vertical --basename dance_60_15

