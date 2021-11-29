import json
import os
import shutil


with open('data/mlb-youtube-segmented.json', 'r') as f:
    data = json.load(f)
    balls = []
    strikes = []
    for k in data:
        labels = data[k]["labels"]
        if labels == ['strike']:
            strikes.append('seg_videos/' + k + '.mp4')
        elif labels == ["ball"]:
            balls.append('seg_videos/' + k + '.mp4')
        else:
            continue



print('starting scan')
directory = 'seg_videos'

for file in os.scandir(directory):
    if file.path in balls:
        shutil.copy(file.path, 'balls')
    elif file.path in strikes:
        shutil.copy(file.path, 'strikes')


