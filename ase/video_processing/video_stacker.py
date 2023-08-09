import argparse
from moviepy.editor import *
from ase.video_processing.video_saver import save_clip
from ase.video_processing.video_cutter import remove_beg_and_end

def stack(videoclip_list, orientation):
    if orientation == 'horizontal':
        with_margin = [vc.margin(right=10, opacity=0.0) for vc in videoclip_list[:-1]]
        with_margin.append(videoclip_list[-1])
        stacked_clip = clips_array([with_margin])
    elif orientation == "vertical":
        with_margin = [vc.margin(bottom=10, opacity=0.0) for vc in videoclip_list[:-1]]
        with_margin.append(videoclip_list[-1])
        stacked_clip = clips_array([[vc] for vc in with_margin])
    return stacked_clip


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

    same_duration_clips_max = make_same_duration_max(cutted_clips, max_duration)
    same_duration_clips_min = make_same_duration_min(cutted_clips, min_duration)
    stacked_clip_max = stack(same_duration_clips_max, orientation)
    stacked_clip_min = stack(same_duration_clips_min, orientation)
    return stacked_clip_max, stacked_clip_min


def stack_sync_from_fname(videofilenames_list, sync_time_list, orientation):
    clip_list = [VideoFileClip(f) for f in videofilenames_list]
    stacked_clip = stack_sync(clip_list, sync_time_list, orientation)
    return stacked_clip


def make_same_duration_max(videoclip_list, max_duration):
    same_duration_clips = []
    for clip_i in videoclip_list:
        if clip_i.duration == max_duration:
            same_duration_clips.append(clip_i)
        else:
            nclip = extend_last_frame(clip_i, max_duration-clip_i.duration)
            same_duration_clips.append(nclip)
    return same_duration_clips


def make_same_duration_min(videoclip_list, min_duration):
    same_duration_clips = []
    for clip_i in videoclip_list:
        if clip_i.duration == min_duration:
            same_duration_clips.append(clip_i)
        else:
            duration_dif = clip_i.duration - min_duration
            nclip = clip_i.cutout(clip_i.duration-duration_dif, clip_i.duration)
            same_duration_clips.append(nclip)
    return same_duration_clips


def extend_last_frame(clip, time):
    dt = 1.0/clip.fps
    last_frame_time = clip.duration-dt*2 #when using just one dt befire it is not working
    last_frame_clip = clip.to_ImageClip(last_frame_time)
    last_frame_clip = last_frame_clip.set_duration(time)
    nclip = concatenate_videoclips([clip, last_frame_clip])
    return nclip


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

    sync_times_start_and_end_list = []
    for i in range(len(args.sync_times_start_and_end) // 2):
        it_idx = 2 * i
        sync_times_start_and_end_list.append(
            (float(args.sync_times_start_and_end[it_idx]), float(args.sync_times_start_and_end[it_idx + 1]))
        )
    stacked_max, stacked_min = stack_sync_from_fname(args.video_files_paths,
                                                             sync_times_start_and_end_list,
                                                             args.orientation)
    dir_name = os.path.dirname(args.video_files_paths[0])
    save_clip(stacked_max, os.path.join(dir_name, "sync_max.mp4"))
    save_clip(stacked_min, os.path.join(dir_name, "sync_min.mp4"))



