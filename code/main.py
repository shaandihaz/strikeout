import tensorflow as tf
import numpy as np
from model import Model, load_frames
from os import listdir

if __name__ == '__main__':
    frames_test = load_frames("../balls_sample/ZWFQJX0RJMQB.mp4", 1000, (500, 500))
    #print(frames_test.shape)

    balls_vid_names = listdir("../balls_sample")
    strikes_vid_names = listdir("../strikes_sample")

    balls_batch = []
    min_frames = 5000
    for vid in balls_vid_names:
        path = "../balls_sample/" + vid
        frames = load_frames(path, 200, (720, 1280))
        # print(frames.shape)
        if frames.shape[0] < min_frames:
            min_frames = frames.shape[0]
        balls_batch.append(frames)

    strikes_batch = []
    for vid in strikes_vid_names:
        path = "../strikes_sample/" + vid
        frames = load_frames(path, 200, (720, 1280))
        # print(frames.shape)
        if frames.shape[0] < min_frames:
            min_frames = frames.shape[0]
        strikes_batch.append(frames)

    # clip videos to be same length (# frames of smallest video in set)
    for i in range(len(balls_batch)):
        if balls_batch[i].shape[0] > min_frames:
            balls_batch[i] = balls_batch[i][:min_frames]
    balls_batch = np.array(balls_batch)
    
    # clip videos to be same length (# frames of smallest video in set)
    for i in range(len(strikes_batch)):
        if strikes_batch[i].shape[0] > min_frames:
            strikes_batch[i] = strikes_batch[i][:min_frames]
    strikes_batch = np.array(strikes_batch)

    n = len(balls_batch) + len(strikes_batch)

    labels = [] # 0 is ball, 1 is strike
    for i in range(len(balls_batch)):
        labels.append([1, 0])
    for i in range(len(strikes_batch)):
        labels.append([0, 1])

    labels = tf.Variable(labels)
    #print(labels)

    #print(balls_batch.shape)
    #print(strikes_batch.shape)
    all_inputs = np.concatenate((balls_batch, strikes_batch))

    all_inputs = tf.Variable(all_inputs)
    #print(all_inputs.shape)

    # shuffle inputs
    indices = tf.Variable(range(all_inputs.shape[0]))
    shuffled = tf.random.shuffle(indices)

    train_inputs = tf.gather(all_inputs, shuffled)
    train_labels = tf.gather(labels, shuffled)

    model = Model()

    num_epochs = 10
    # acc, loss = model.train(train_inputs, train_labels, num_epochs)
    # print("Loss: " + str(loss) + ", Accuracy: " + str(acc))
    model.fit(train_inputs, train_labels, epochs=num_epochs)