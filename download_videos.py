import os
import json
import string
import random
import subprocess


save_dir = 'full_videos'
with open('data/mlb-youtube-segmented.json', 'r') as f:
    data = json.load(f)
    print(len(data))
    
    for entry in data:
        yturl = data[entry]['url']
        ytid = yturl.split('=')[-1]

        print(os.path.join(save_dir, ytid+'.mp4'))
        
        if os.path.exists(os.path.join(save_dir, ytid+'.mp4')):
            continue

        cmd = 'yt-dlp -f mp4 '+yturl+' -o '+os.path.join(ytid+'.mp4')
        os.system(cmd)
