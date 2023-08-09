cd ../..
export PYTHONPATH=/home/erick/MotionProjs/ASE_MODS/lib:$PYTHONPATH


#python ase/video_processing/video_stacker_with_duration_change.py --video_files_paths /home/erick/Videos/exp_vids/plus_motions/dance/019_Relaxed_3_mirror_x_1_0/gt.avi /home/erick/Videos/exp_vids/plus_motions/dance/019_Relaxed_3_mirror_x_1_0/H+2C.avi /home/erick/Videos/exp_vids/plus_motions/dance/019_Relaxed_3_mirror_x_1_0/H+4C.avi --sync_times_start_and_end 4.22 21.99 4.09 16.74 4.06 17.22 --orientation vertical

python ase/video_processing/video_stacker.py --video_files_paths /home/erick/Videos/exp_vids/plus_motions/dance/019_Relaxed_3_mirror_x_1_0/gt.avi /home/erick/Videos/exp_vids/plus_motions/dance/019_Relaxed_3_mirror_x_1_0/H+2C.avi /home/erick/Videos/exp_vids/plus_motions/dance/019_Relaxed_3_mirror_x_1_0/H+4C.avi --sync_times_start_and_end 4.22 21.99 4.09 16.74 4.06 17.22 --orientation vertical
