import argparse
from moviepy.editor import *
from ase.video_processing.video_stacker import stack, save_clip
from ase.video_processing.video_cutter import remove_beg_and_end
from ase.video_processing.video_duration_changer import change_duration

def stack_sync(videoclip_list, sync_times_start_and_end_list, orientation):
    #get the earliest time to know how much to cut from others
    cutted_clips = []
    max_duration = -1
    min_duration = None
    for i in range(len(videoclip_list)):
        c_clip = videoclip_list[i]
        c_beg_rem, c_end_rem = sync_times_start_and_end_list[i]
        c_clip = remove_beg_and_end(c_clip, c_beg_rem, c_end_rem)
        cutted_clips.append(c_clip)
        if c_clip.duration > max_duration:
            max_duration = c_clip.duration
        if min_duration is None or c_clip.duration < min_duration:
            min_duration = c_clip.duration

    same_duration_clips_max = make_all_same_duration(cutted_clips, max_duration)
    stacked_clip_max = stack(same_duration_clips_max, orientation)
    return stacked_clip_max


def make_all_same_duration(videoclip_list, desired_duration):
    same_duration_clips = []
    for clip_i in videoclip_list:
        modified_clip = change_duration(clip_i, desired_duration)
        same_duration_clips.append(modified_clip)
    return same_duration_clips

def stack_sync_from_fname(videofilenames_list, sync_time_list, orientation):
    clip_list = [VideoFileClip(f) for f in videofilenames_list]
    stacked_clip = stack_sync(clip_list, sync_time_list, orientation)
    return stacked_clip


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_files_paths', nargs='+', default=[],
                        help='paths to the video files',
                        required=True)
    parser.add_argument('--sync_times_start_and_end', nargs='+', default=[],
                        help='points of time where to sync videos',
                        required=True)
    parser.add_argument('--orientation',
                        type=str,
                        choices=['vertical', 'horizontal'],
                        help='The direction in which the videos are stacked',
                        required=True)

    args = parser.parse_args()



    assert len(args.sync_times_start_and_end) == 2 * len(args.video_files_paths)

    sync_times_start_and_end_list = []
    for i in range(len(args.sync_times_start_and_end) // 2):
        it_idx = 2 * i
        sync_times_start_and_end_list.append(
            (float(args.sync_times_start_and_end[it_idx]), float(args.sync_times_start_and_end[it_idx + 1]))
        )
    stacked_max = stack_sync_from_fname(args.video_files_paths,
                                                             sync_times_start_and_end_list,
                                                             args.orientation)
    dir_name = os.path.dirname(args.video_files_paths[0])
    save_clip(stacked_max, os.path.join(dir_name, "sync_max_dch.mp4"))