from moviepy.editor import *
import json
import os
import shutil

dir = "balls"

for file in os.scandir(dir):
    clip = VideoFileClip(file.path)
    clip = clip.subclip(0,3)
    clip.write_videofile('cut_balls/' + file.path.split('/')[1])
    

