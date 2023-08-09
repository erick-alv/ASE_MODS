from moviepy.editor import *

def save_clip(clip, save_name):
    clip = clip.without_audio()
    clip.write_videofile(save_name)