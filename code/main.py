import tensorflow as tf
import numpy as np
import random
import cv2
from model import Model
from os import listdir

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
    return np.array(frames)

def load_videos(video_paths):
    video_batch = []
    good_indices = []
    tot_frames = 0
    i = 0
    for j in range(len(video_paths)):
        print("Loading Video " + str(i))
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
        if vid.startswith("../b"):
            indices.append(0) # ball
        else:
            indices.append(1) # strike
    return tf.one_hot(indices, 2)
    
    

if __name__ == '__main__':
    # frames_test = load_frames("../balls_sample/ZWFQJX0RJMQB.mp4", 1000, (500, 500))
    #print(frames_test.shape)

    balls_dir = "../balls/"
    strikes_dir = "../strikes/"

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

    # iteratively grab batches to train
    for i in range(0, len(train_names), step_size):
        video_batch = train_names[i:i+step_size]
        labels = get_labels(video_batch)
        # print(labels)
        videos, avg_frames, good_indices = load_videos(video_batch)
        avg_frames = 140
        labels = tf.gather(labels, good_indices)
        assert(labels.shape[0] == len(videos))
        print(avg_frames)
        normalize_frames(videos, avg_frames)
        # for v in videos:
        #     print(v.shape)
        videos = np.array(videos)
        print(videos.shape)
        print(labels.shape)
        model.train(videos, labels, num_epochs)

    # iteratively grab batches to test
    for i in range(0, len(test_names), step_size):
        video_batch = test_names[i:i+step_size]
        labels = get_labels(video_batch)
        # print(labels)
        videos, min_frames = load_videos(video_batch)
        print(min_frames)
        normalize_frames(videos, avg_frames)
        # for v in videos:
        #     print(v.shape)
        videos = np.array(videos)
        model.test(videos, labels)




    # labels = [] # 0 is ball, 1 is strike
    # for i in range(len(balls_batch)):
    #     labels.append([1, 0])
    # for i in range(len(strikes_batch)):
    #     labels.append([0, 1])

    # labels = tf.Variable(labels)

    # balls_batch = []
    # min_frames = 5000
    # i = 1
    # for vid in balls_vid_names:
    #     print("Ball " + str(i))
    #     path = balls_dir + vid
    #     frames = load_frames(path, 200, (720, 1280))
    #     # print(frames.shape)
    #     if frames.shape[0] < min_frames:
    #         min_frames = frames.shape[0]
    #     balls_batch.append(frames)
    #     i += 1

    # strikes_batch = []
    # i = 1
    # for vid in strikes_vid_names:
    #     print("Strike " + str(i))
    #     path = strikes_dir + vid
    #     frames = load_frames(path, 200, (720, 1280))
    #     # print(frames.shape)
    #     if frames.shape[0] < min_frames:
    #         min_frames = frames.shape[0]
    #     strikes_batch.append(frames)
    #     i += 1

    # # clip videos to be same length (# frames of smallest video in set)
    # for i in range(len(balls_batch)):
    #     if balls_batch[i].shape[0] > min_frames:
    #         balls_batch[i] = balls_batch[i][:min_frames]
    # balls_batch = np.array(balls_batch)
    
    # # clip videos to be same length (# frames of smallest video in set)
    # for i in range(len(strikes_batch)):
    #     if strikes_batch[i].shape[0] > min_frames:
    #         strikes_batch[i] = strikes_batch[i][:min_frames]
    # strikes_batch = np.array(strikes_batch)

    # video_batch = np.array(video_batch)

    # n = len(balls_batch) + len(strikes_batch)

    # labels = [] # 0 is ball, 1 is strike
    # for i in range(len(balls_batch)):
    #     labels.append([1, 0])
    # for i in range(len(strikes_batch)):
    #     labels.append([0, 1])

    # labels = tf.Variable(labels)
    #print(labels)

    #print(balls_batch.shape)
    #print(strikes_batch.shape)
    # all_inputs = np.concatenate((balls_batch, strikes_batch))

    # all_inputs = tf.Variable(all_inputs)
    #print(all_inputs.shape)

    # shuffle inputs
    # indices = tf.Variable(range(all_inputs.shape[0]))
    # shuffled = tf.random.shuffle(indices)

    # train_inputs = tf.gather(all_inputs, shuffled)
    # train_labels = tf.gather(labels, shuffled)

    # model = Model()

    # num_epochs = 10
    # acc, loss = model.train(train_inputs, train_labels, num_epochs)
    # print("Loss: " + str(loss) + ", Accuracy: " + str(acc))
    # model.fit(train_inputs, train_labels, epochs=num_epochs)