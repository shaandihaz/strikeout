import tensorflow as tf
import numpy as np
import random
import cv2
from model import Model
from os import listdir
import sys

def load_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        #print(frame.shape)
        if not ret:
            break
        frame = cv2.resize(frame, (360, 640))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray.astype('float32')
        gray /= 255
        #print(gray.shape)
        frames.append(gray)
    cap.release()
    return np.array(frames)

def load_videos(video_paths):
    video_batch = []
    good_indices = []
    tot_frames = 0
    i = 0
    for j in range(len(video_paths)):
        #print("Loading Video " + str(i))
        frames = load_frames(video_paths[j])
        if frames.shape[0] == 0:
            # skip corrupted files
            continue
        # print(frames.shape)
        good_indices.append(j)
        tot_frames += frames.shape[0]
        video_batch.append(frames)
        i += 1
    avg_frames = int(tot_frames / i)
    return video_batch, avg_frames, good_indices

def normalize_frames(video_batch, avg_frames):
    for i in range(len(video_batch)):
        #print(vid.shape)
        numf = video_batch[i].shape[0]
        #print(numf)
        if numf < avg_frames:
            video_batch[i] = pad_frames(video_batch[i], avg_frames, (360, 640))
        elif numf > avg_frames:
            video_batch[i] = trim_frames(video_batch[i], avg_frames)
        #print(vid.shape)

def trim_frames(video, min_frames):
    if video.shape[0] > min_frames:
        return video[:min_frames]

def pad_frames(video, num_frames, shape):
    numAdd = num_frames - video.shape[0]
    return np.pad(video, ((0, numAdd), (0,0), (0,0)))

def get_labels(video_batch):
    # Get one-hot tensors as label for each video in batch
    indices = []
    for vid in video_batch:
        if vid.startswith("../cut_b"):
            indices.append(0) # ball
        else:
            indices.append(1) # strike
    return tf.one_hot(indices, 2)
    
    

if __name__ == '__main__':
    # frames_test = load_frames("../balls_sample/ZWFQJX0RJMQB.mp4", 1000, (500, 500))
    #print(frames_test.shape)

    args = sys.argv[1:]



    balls_dir = "../cut_balls/"
    strikes_dir = "../cut_strikes/"

    balls_vid_names = listdir(balls_dir)
    strikes_vid_names = listdir(strikes_dir)
    balls_vid_names = [balls_dir + n for n in balls_vid_names] # prepend directory
    strikes_vid_names = [strikes_dir + n for n in strikes_vid_names] # prepend directory
    all_vid_names = balls_vid_names + strikes_vid_names


    # since entire dataset does not fit into memory, successively
    # load batches of videos into model, calling model.fit each time
    
    # shuffle all names
    random.shuffle(all_vid_names)
    #print(all_vid_names)

    ind = int(len(all_vid_names) * 0.8)
    train_names = all_vid_names[:ind]
    test_names = all_vid_names[ind:]

    # initialize model
    step_size = 50
    num_epochs = 1 # number of epochs per train/test batch
    batch_size = step_size
    model = Model(batch_size)

    if args != []:
        model.load_weights('./checkpoints/my_chkpt')
        print('loaded')



    # iteratively grab batches to train
    numb = 0
    tot = int(len(train_names) / step_size)
    avg_loss = 0
    avg_acc = 0
    train_losses = []
    train_accs = []
    for ep in range(1):
        print("Train Epoch " + str(ep))
        for i in range(0, len(train_names), step_size):
            print("Batch " + str(numb) + "/" + str(tot))
            video_batch = train_names[i:i+step_size]
            labels = get_labels(video_batch)
            # print(labels)
            videos, avg_frames, good_indices = load_videos(video_batch)
            avg_frames = 140
            labels = tf.gather(labels, good_indices)
            assert(labels.shape[0] == len(videos))
            normalize_frames(videos, avg_frames)
            # for v in videos:
            #     print(v.shape)
            videos = np.array(videos)
            acc, loss = model.train(videos, labels, num_epochs)
            train_losses.append(loss)
            train_accs.append(acc)
            numb += 1
        print("Epoch avg loss: " + str(sum(train_losses) / len(train_losses)) + ", avg acc: " + str(sum(train_accs) / len(train_accs)))

    print("Train avg loss: " + str(sum(train_losses) / len(train_losses)) + ", avg acc: " + str(sum(train_accs) / len(train_accs)))

    model.save_weights('./checkpoints/my_chkpt')

    model.visualize_loss(train_losses)

    # iteratively grab batches to test
    numb = 0
    avg_loss = 0
    avg_acc = 0
    test_losses = []
    test_accs = []
    tot = int(len(test_names) / step_size)
    for i in range(0, len(test_names), step_size):
        print("Batch " + str(numb) + "/" + str(tot))
        video_batch = test_names[i:i+step_size]
        labels = get_labels(video_batch)
        # print(labels)
        videos, avg_frames, good_indices = load_videos(video_batch)
        avg_frames = 140
        labels = tf.gather(labels, good_indices)
        normalize_frames(videos, avg_frames)
        # for v in videos:
        #     print(v.shape)
        videos = np.array(videos)
        acc, loss = model.test(videos, labels)
        test_accs.append(acc)
        test_losses.append(loss)
        numb += 1
    print("Test avg loss: " + str(sum(test_losses) / len(test_losses)) + ", avg acc: " + str(sum(test_accs) / len(test_accs)))
