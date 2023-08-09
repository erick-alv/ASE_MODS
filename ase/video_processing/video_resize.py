import argparse
from moviepy.editor import *
from ase.video_processing.video_saver import save_clip

def resize(clip, width, height):
    return clip.resize((width, height))

def resize_from_fname(videofilename, width, height):
    clip = VideoFileClip(videofilename)
    resized = resize(clip, width, height)
    return resized

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_files_paths', nargs='+', default=[],
                        help='paths to the video files',
                        required=True)
    parser.add_argument('--width',
                        type=int,
                        help='new width',
                        required=True)
    parser.add_argument('--height',
                        type=int,
                        help='new height',
                        required=True)
    args = parser.parse_args()
    width = int(args.width)
    height = int(args.height)
    for i in range(len(args.video_files_paths)):
        fname = args.video_files_paths[i]
        resized = resize_from_fname(fname, width, height)
        new_fname = fname[:-4] + "_resized.mp4"
        save_clip(resized, new_fname)