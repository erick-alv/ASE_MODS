import argparse
from ase.video_processing.video_stacker import stack
from moviepy.editor import *

def stack_and_frame(videoclip_list, sync_time_list, orientation, savename):
    framesClips = []
    for i in range(len(videoclip_list)):
        clip = videoclip_list[i]
        time = sync_time_list[i]
        frame = clip.get_frame(time)
        framesClips.append(ImageClip(frame))
    stacked_clip = stack(framesClips, orientation)
    stacked_clip.save_frame(savename)


def stack_and_frame_from_fname(videofilenames_list, sync_time_list, orientation, savename):
    clip_list = [VideoFileClip(f) for f in videofilenames_list]
    stack_and_frame(clip_list, sync_time_list, orientation, savename)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_files_paths', nargs='+', default=[],
                        help='paths to the video files',
                        required=True)
    parser.add_argument('--sync_times', nargs='+', default=[],
                        help='points of time where to sync videos',
                        required=True)
    parser.add_argument('--orientation',
                        type=str,
                        choices=['vertical', 'horizontal'],
                        help='The direction in which the videos are stacked',
                        required=True)

    parser.add_argument('--basename',
                        type=str,
                        default="",
                        help='The basename')

    args = parser.parse_args()
    dir_name = os.path.dirname(args.video_files_paths[0])

    assert len(args.sync_times) % len(args.video_files_paths) == 0
    n_vids = len(args.video_files_paths)
    n_pr = len(args.sync_times) // n_vids
    for n in range(n_pr):
        beg_idx = n_vids*n
        end_idx = n_vids*n+n_vids
        sync_times_str = args.sync_times[beg_idx: end_idx]
        sync_times = [float(v) for v in sync_times_str]

        if len(args.basename) > 0:
            filename = args.basename + "_"
        else:
            filename = "stacked_at_"
        filename += "_".join(sync_times_str)
        filename += ".png"


        filename = os.path.join(dir_name, filename)
        stack_and_frame_from_fname(args.video_files_paths, sync_times, args.orientation, filename)
