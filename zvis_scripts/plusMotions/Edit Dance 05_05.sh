cd ../..
export PYTHONPATH=/home/erick/MotionProjs/ASE_MODS/lib:$PYTHONPATH


#python ase/video_processing/stack_frame.py --video_files_paths /home/erick/Videos/exp_vids/plus_motions/dance/05_05/gt.avi /home/erick/Videos/exp_vids/plus_motions/dance/05_05/H+2C.avi /home/erick/Videos/exp_vids/plus_motions/dance/05_05/H+4C.avi --sync_times 2.67 2.80 2.09 --orientation vertical --basename dance_05_05

#python ase/video_processing/stack_frame.py --video_files_paths /home/erick/Videos/exp_vids/plus_motions/dance/05_05/gt.avi /home/erick/Videos/exp_vids/plus_motions/dance/05_05/H+2C.avi /home/erick/Videos/exp_vids/plus_motions/dance/05_05/H+4C.avi --sync_times 4.25 3.54 2.77 --orientation vertical --basename dance_05_05

python ase/video_processing/stack_frame.py --video_files_paths /home/erick/Videos/exp_vids/plus_motions/dance/05_05/gt.avi /home/erick/Videos/exp_vids/plus_motions/dance/05_05/H+2C.avi /home/erick/Videos/exp_vids/plus_motions/dance/05_05/H+4C.avi --sync_times 5.16 4.28 4.12 --orientation vertical --basename dance_05_05
