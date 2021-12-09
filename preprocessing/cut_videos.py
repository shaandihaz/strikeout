from moviepy.editor import *
import json
import os
import shutil

dir_balls = "../balls"
dir_strikes = "../strikes"

print(os.getcwd())
balls_done = set(os.listdir("../cut_balls"))
strikes_done = set(os.listdir("../cut_strikes"))
print(balls_done)
print(strikes_done)

for file in os.listdir(dir_balls):
    if file not in balls_done:
        try:
            clip = VideoFileClip(dir_balls + "/" + file)
            clip = clip.subclip(0,3)
            clip.write_videofile('../cut_balls/' + file)
        except KeyError:
            print(file)

for file in os.listdir(dir_strikes):
    if file not in strikes_done:
        try:
            clip = VideoFileClip(dir_strikes + "/" + file)
            clip = clip.subclip(0,3)
            clip.write_videofile('../cut_strikes/' + file)
        except KeyError:
            print(file)
    

