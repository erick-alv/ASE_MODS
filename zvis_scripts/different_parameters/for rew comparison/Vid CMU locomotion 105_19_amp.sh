cd ../../..
export PYTHONPATH=/home/erick/MotionProjs/ASE_MODS/lib:$PYTHONPATH

#python ase/video_processing/stack_frame.py --video_files_paths /home/erick/Videos/exp_vids/different_parameters/for_rew/loc_105_19_amp/gt.avi /home/erick/Videos/exp_vids/different_parameters/for_rew/loc_105_19_amp/altto_rt1.avi /home/erick/Videos/exp_vids/different_parameters/for_rew/loc_105_19_amp/altto_rt2.avi --sync_times 3.93 4.03 3.38 --orientation vertical --basename rew_rt1_vs_rt2

#python ase/video_processing/stack_frame.py --video_files_paths /home/erick/Videos/exp_vids/different_parameters/for_rew/loc_105_19_amp/gt.avi /home/erick/Videos/exp_vids/different_parameters/for_rew/loc_105_19_amp/altto_rt1.avi /home/erick/Videos/exp_vids/different_parameters/for_rew/loc_105_19_amp/altto_rt2.avi --sync_times 5.28 5.77 4.9 --orientation vertical --basename rew_rt1_vs_rt2

#python ase/video_processing/stack_frame.py --video_files_paths /home/erick/Videos/exp_vids/different_parameters/for_rew/loc_105_19_amp/gt.avi /home/erick/Videos/exp_vids/different_parameters/for_rew/loc_105_19_amp/altto_rt1.avi /home/erick/Videos/exp_vids/different_parameters/for_rew/loc_105_19_amp/altto_rt2.avi --sync_times 7.96 9.48 7.57 --orientation vertical --basename rew_rt1_vs_rt2

#python ase/video_processing/stack_frame.py --video_files_paths /home/erick/Videos/exp_vids/different_parameters/for_rew/loc_105_19_amp/gt.avi /home/erick/Videos/exp_vids/different_parameters/for_rew/loc_105_19_amp/altto_rt1.avi /home/erick/Videos/exp_vids/different_parameters/for_rew/loc_105_19_amp/altto_rt2.avi --sync_times 9.64 11.74 8.77 --orientation vertical --basename rew_rt1_vs_rt2

#TODO: Yes, but since opted to end before; the video of fallen seems to be smaller --> have to see times again.
python ase/video_processing/video_stacker_with_duration_change.py --video_files_paths /home/erick/Videos/exp_vids/different_parameters/for_rew/loc_105_19_amp/gt.avi /home/erick/Videos/exp_vids/different_parameters/for_rew/loc_105_19_amp/altto_rt1.avi /home/erick/Videos/exp_vids/different_parameters/for_rew/loc_105_19_amp/altto_rt2.avi --sync_times_start_and_end 5.28 12.61 5.77 11.87 4.9 12.16 --orientation vertical






