from moviepy.editor import *
import argparse

def save_frame_at(clip, time_t, save_name):
    assert time_t <= clip.duration
    clip.save_frame(save_name, t=time_t)


def save_frame_at_from_file(videofilename, time_t, save_name):
    clip = VideoFileClip(videofilename)
    save_frame_at(clip, time_t, save_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file_path', type=str,
                        help='paths to the video files',
                        required=True)
    parser.add_argument('--frame_times', nargs='+', default=[],
                        help='times of the respective frames',
                        required=True)
    args = parser.parse_args()

    for fr_time in args.frame_times:
        frame_save_name = args.video_file_path[:-4] + f"_at{fr_time}.png"
        fr_time_f = float(fr_time)
        save_frame_at_from_file(args.video_file_path, fr_time_f, frame_save_name)
