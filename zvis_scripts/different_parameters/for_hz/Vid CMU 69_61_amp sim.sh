cd ../../..
export PYTHONPATH=/home/erick/MotionProjs/ASE_MODS/lib:$PYTHONPATH


python ase/video_processing/video_stacker_with_duration_change.py --video_files_paths /home/erick/Videos/exp_vids/different_parameters/for_hz/69_61_amp_sim_side_walking/gt.avi /home/erick/Videos/exp_vids/different_parameters/for_hz/69_61_amp_sim_side_walking/rt2_pd_alt_36.avi /home/erick/Videos/exp_vids/different_parameters/for_hz/69_61_amp_sim_side_walking/rt2_pd_alt_72.avi --sync_times_start_and_end 2.06 4.86 2.06 4.86 2.06 8.03 --orientation vertical

python ase/video_processing/video_stacker.py --video_files_paths /home/erick/Videos/exp_vids/different_parameters/for_hz/69_61_amp_sim_side_walking/gt.avi /home/erick/Videos/exp_vids/different_parameters/for_hz/69_61_amp_sim_side_walking/rt2_pd_alt_36.avi /home/erick/Videos/exp_vids/different_parameters/for_hz/69_61_amp_sim_side_walking/rt2_pd_alt_72.avi --sync_times_start_and_end 2.06 4.86 2.06 4.86 2.06 8.03 --orientation vertical