import argparse
from moviepy.editor import *

def get_size_from_fname(videofilenames_list):
    clip_list = [VideoFileClip(f) for f in videofilenames_list]
    for clip in clip_list:
        print(clip.size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_files_paths', nargs='+', default=[],
                        help='paths to the video files',
                        required=True)
    args = parser.parse_args()
    get_size_from_fname(args.video_files_paths)