import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras import Model
from tensorflow.keras.applications import InceptionV3

from os import listdir


class Model(tf.keras.Model):
    def __init__(self):
        """
        The Model class predicts the outcome of the pitch from the windup clip.
        """
        super(Model, self).__init__()

        self.batch_size = 100
        self.learning_rate = 0.001

        self.pipeline = []
        conv1 = Conv2D(64, 3, 1, padding="same", activation="relu")
        conv2 = Conv2D(64, 3, 1, padding="same", activation="relu")
        max1 = MaxPool2D(2)
        conv3 = Conv2D(128, 3, 1, padding="same", activation="relu")
        conv4 = Conv2D(128, 3, 1, padding="same", activation="relu")
        max2 = MaxPool2D(2)
        conv5 = Conv2D(256, 3, 1, padding="same", activation="relu")
        conv6 = Conv2D(256, 3, 1, padding="same", activation="relu")
        max3 = MaxPool2D(2)
        flat = Flatten()
        drop = Dropout(0.2)
        dense1 = Dense(15, activation="softmax")
        self.architecture = [conv1, conv2, max1, conv3, conv4, max2, conv5, conv6, max3, flat, drop, dense1]

    def call(self, inputs):
        """
        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: dont worry about it
        """
        # Take in (batch_size x num_frames x width x height) video clips
        # and (batch_size x 1) labels
        # Output (batch_size x 2) probs matrix
        for layer in self.pipeline:
            inputs = layer(inputs)
        return inputs
        

    def loss(self, logits, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        
        NOTE: You have to use np.reduce_mean and not np.reduce_sum when calculating your loss

        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """

        #TODO: Fill in
        #We recommend using tf.keras.losses.sparse_categorical_crossentropy
        #https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy
        # probs is actually logits
        loss = tf.keras.losses.binary_crossentropy(labels, logits)
        return tf.reduce_mean(loss)

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT
        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    # clip train inputs and labels to multiple of window size

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    num_batches = 1
    avg_loss = 0
    avg_acc = 0
    for i in range(0, train_inputs.shape[0], model.batch_size):
        batch = train_inputs[i:i + model.batch_size]
        label_batch = train_labels[i:i + model.batch_size]
        # Implement backprop:
        with tf.GradientTape() as tape:
            logits = model.call(batch) # this calls the call function conveniently
            loss = model.loss(logits, label_batch)
            acc = model.accuracy(logits, label_batch)
            avg_loss += loss
            avg_acc += acc

        # The keras Model class has the computed property trainable_variables to conveniently
        # return all the trainable variables you'd want to adjust based on the gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        num_batches += 1

    avg_loss /= num_batches
    avg_acc /= num_batches
    return avg_loss, avg_acc

def test(model, test_inputs, test_labels):
    # fix array indices error
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """
    #NOTE: Ensure a correct perplexity formula (different from raw loss)
    avg_loss = 0.0
    num_batches = 0
    for i in range(0, test_inputs.shape[0], model.batch_size):
        num_batches += 1
        batch = test_inputs[i:i + model.batch_size]
        label_batch = test_labels[i:i + model.batch_size]
        probs = model.call(batch)
        # Get the average loss for each row in probs (2d size batch_size x vocab_size)
        # Add to overall average loss for each batch
        avg_loss += model.loss(probs, label_batch)
    
    avg_loss = avg_loss / num_batches
    return avg_loss

    

def load_frames(video_path, max_frames, size):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        #print(frame.shape)
        if not ret:
            break
        #frame = cv2.resize(frame, size)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #print(gray.shape)
        frames.append(gray)
        if len(frames) == max_frames:
            break
    return np.array(frames)


def main():
    frames_test = load_frames("../balls_sample/ZWFQJX0RJMQB.mp4", 1000, (500, 500))
    print(frames_test.shape)

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
        labels.append(0)
    for i in range(len(strikes_batch)):
        labels.append(1)

    labels = tf.Variable(labels)
    print(labels)

    print(balls_batch.shape)
    print(strikes_batch.shape)
    all_inputs = np.concatenate((balls_batch, strikes_batch))

    all_inputs = tf.Variable(all_inputs)
    print(all_inputs.shape)

    # shuffle inputs
    indices = tf.Variable(range(len(all_inputs)))
    shuffled = tf.random.shuffle(indices)

    train_inputs = tf.gather(all_inputs, shuffled)
    train_labels = tf.gather(labels, shuffled)

    model = Model()

    num_epochs = 1
    for i in range(num_epochs):
        loss, acc = train(model, train_inputs, train_labels)
        print("Epoch: " + i)
        print("Loss: " + loss + ", Accuracy: " + acc)

if __name__ == '__main__':
    main()
