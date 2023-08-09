cd ../..
export PYTHONPATH=/home/erick/MotionProjs/ASE_MODS/lib:$PYTHONPATH

python ase/video_processing/video_stacker_with_duration_change.py --video_files_paths /home/erick/Videos/exp_vids/plus_motions/jump/dataset-2_walk-turn-right_normal_030/gt.avi /home/erick/Videos/exp_vids/plus_motions/jump/dataset-2_walk-turn-right_normal_030/H+2C.avi /home/erick/Videos/exp_vids/plus_motions/jump/dataset-2_walk-turn-right_normal_030/H+4C.avi  --sync_times_start_and_end 2.06 6.16 1.975 4.8 2.06 4.96 --orientation vertical

python ase/video_processing/video_stacker.py --video_files_paths /home/erick/Videos/exp_vids/plus_motions/jump/dataset-2_walk-turn-right_normal_030/gt.avi /home/erick/Videos/exp_vids/plus_motions/jump/dataset-2_walk-turn-right_normal_030/H+2C.avi /home/erick/Videos/exp_vids/plus_motions/jump/dataset-2_walk-turn-right_normal_030/H+4C.avi  --sync_times_start_and_end 2.06 6.16 1.975 4.8 2.06 4.96 --orientation vertical
