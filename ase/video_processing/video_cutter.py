from moviepy.editor import *
from ase.video_processing.video_saver import save_clip
import argparse

def remove_beg_and_end(clip, beg_remove, end_remove):
    #removes begin
    time_to_remove = clip.duration - end_remove
    e_clip = clip.cutout(0, beg_remove)
    e_duration = e_clip.duration
    e_clip = e_clip.cutout(e_duration-time_to_remove, e_duration)
    return e_clip


def remove_beg_and_end_from_fname(videofilename, beg_remove, end_remove):
    clip = VideoFileClip(videofilename)
    e_clip = remove_beg_and_end(clip, beg_remove, end_remove)
    return e_clip


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_files_paths', nargs='+', default=[],
                        help='paths to the video files',
                        required=True)
    parser.add_argument('--del_times', nargs='+', default=[],
                        help='times in seconds to delete from the file; 1 value for the beginnging and 1 value for the end',
                        required=True)
    args = parser.parse_args()


    assert len(args.del_times) == 2*len(args.video_files_paths)

    del_times_list = []
    for i in range(len(args.del_times)//2):
        it_idx = 2*i
        del_times_list.append(
            (float(args.del_times[it_idx]), float(args.del_times[it_idx+1]))
        )

    for i in range(len(args.video_files_paths)):
        fname = args.video_files_paths[i]
        beg_time_remove = del_times_list[i][0]
        end_time_remove = del_times_list[i][1]
        cutted_clip = remove_beg_and_end_from_fname(fname, beg_time_remove, end_time_remove)
        new_fname = fname[:-4] + "_cut.mp4"
        save_clip(cutted_clip, new_fname)

