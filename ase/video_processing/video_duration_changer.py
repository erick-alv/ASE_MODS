import argparse
from ase.video_processing.video_stacker import stack
from moviepy.editor import *

def change_duration(videoclip, desired_duration):
    if desired_duration == videoclip.duration:
        return videoclip
    else:
        speed_factor = videoclip.duration / desired_duration
        #modified = videoclip.set_duration(desired_duration)
        modified = videoclip.fx(vfx.speedx, speed_factor)
        return modified


def change_duration_from_fname(videofilename, desired_duration, savename=None):
    clip = VideoFileClip(videofilename)
    modified_clip = change_duration(clip, desired_duration)
    if savename is not None:
        modified_clip.write_videofile(savename)



if __name__ == "__main__":
    vid_name = "/home/erick/Videos/exp_vids/saludos PD, rt2, AltPar, (72Hz) 233.avi"
    out_name = vid_name[:-4] + "_modified.mp4"
    change_duration_from_fname(vid_name, 16.0, out_name)

