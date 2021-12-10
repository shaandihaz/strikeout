import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from tensorflow.keras.metrics import binary_accuracy


class Model(tf.keras.Model):
    def __init__(self, batch_size):
        """
        The Model class predicts the outcome of the pitch from the windup clip.
        """
        super(Model, self).__init__()

        self.batch_size = batch_size
        self.learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.pipeline = tf.keras.Sequential()

        self.pipeline.add(Conv2D(64, 3, 1, padding="same", activation="relu"))
        self.pipeline.add(Conv2D(64, 3, 1, padding="same", activation="relu"))
        self.pipeline.add(MaxPool2D(2))
        self.pipeline.add(Conv2D(128, 3, 1, padding="same", activation="relu"))
        self.pipeline.add(Conv2D(128, 3, 1, padding="same", activation="relu"))
        self.pipeline.add(MaxPool2D(2))
        self.pipeline.add(Conv2D(256, 3, 1, padding="same", activation="relu"))
        self.pipeline.add(Conv2D(256, 3, 1, padding="same", activation="relu"))
        self.pipeline.add(MaxPool2D(2))
        self.pipeline.add(Flatten())
        self.pipeline.add(Dropout(0.2))
        self.pipeline.add(Dense(2, activation="softmax"))
        
        self.pipeline.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=self.optimizer, metrics=[binary_accuracy])

    def train(self, inputs, labels, num_epochs):
        result = self.pipeline.fit(inputs, labels, epochs=num_epochs)
        accuracy = sum(result.history['binary_accuracy']) / len(result.history['binary_accuracy'])
        loss = sum(result.history['loss']) / len(result.history['loss'])
        return accuracy, loss

    # def train_step(self, data):
    #     x, y = data
    #     # x is shape (batch_size x num_frames x width x height)
    #     # want to call pipeline on each frame (3d tensor) for each video and average
    #     # results across frame, then returning this average for each video in batch.
    #     with tf.GradientTape() as tape:
    #         y_pred = []
    #         for i in range(x.shape[0]):
    #             vid = x[i] # the 3d (num_frames x width x height) video tensor
    #             preds = self.pipeline(vid) # should be (num_frames x 2) probs tensor
    #             avg_preds = tf.reduce_mean(preds, axis=0)
    #             y_pred.append(avg_preds)
    #         y_pred = tf.Variable(y_pred)
    #         loss = self.pipeline.compiled_loss(y, y_pred)

    #     gradients = tape.gradients(loss, self.pipeline.trainable_variables)
    #     self.pipeline.optimizer.apply_gradients(zip(gradients, self.pipeline.trainable_variables))
    #     self.pipeline.compiled_metrics.update_state(y, y_pred)
    #     return {m.name: m.result() for m in self.pipeline.metrics}

    def test(self, inputs, labels):
        result = self.pipeline.evaluate(inputs, labels)
        accuracy = result[1]
        loss = result[0]
        return accuracy, loss

    def visualize_loss(self, losses): 
        """
        Uses Matplotlib to visualize the losses of our model.
        :param losses: list of loss data stored from train. Can use the model's loss_list 
        field 
        :return: doesn't return anything, a plot should pop-up 
        """
        x = [i for i in range(len(losses))]
        plt.plot(x, losses)
        plt.title('Loss per batch')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.show()

    # def call(self, inputs):
    #     """
    #     :param inputs: word ids of shape (batch_size, window_size)
    #     :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
    #     :return: dont worry about it
    #     """
    #     # Take in (batch_size x num_frames x width x height) video clips
    #     # and (batch_size x 1) labels
    #     # Output (batch_size x 2) probs matrix
    #     for layer in self.pipeline:
    #         inputs = layer(inputs)
    #     return inputs
        

    # def loss(self, probs, labels):
    #     """
    #     Calculates average cross entropy sequence to sequence loss of the prediction
        
    #     NOTE: You have to use np.reduce_mean and not np.reduce_sum when calculating your loss

    #     :param probs: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
    #     :param labels: matrix of shape (batch_size, window_size) containing the labels
    #     :return: the loss of the model as a tensor of size 1
    #     """
    #     #print(labels)
    #     print(probs.shape)
    #     # logits should be batch_size x 15
    #     loss = tf.keras.losses.binary_crossentropy(labels, probs)
    #     return tf.reduce_mean(loss)

    # def accuracy(self, probs, labels):
    #     """
    #     Calculates the model's prediction accuracy by comparing
    #     logits to correct labels – no need to modify this.
        
    #     :param probs: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
    #     containing the result of multiple convolution and feed forward layers
    #     :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

    #     NOTE: DO NOT EDIT
        
    #     :return: the accuracy of the model as a Tensor
    #     """
    #     y_pred = tf.argmax(probs, 1)
    #     return tf.keras.metrics.binary_accuracy(labels, y_pred)


# def train(model, train_inputs, train_labels):
#     """
#     Runs through one epoch - all training examples.

#     :param model: the initilized model to use for forward and backward pass
#     :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
#     :param train_labels: train labels (all labels for training) of shape (num_labels,)
#     :return: None
#     """
#     # clip train inputs and labels to multiple of window size

#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

#     num_batches = 1
#     avg_loss = 0
#     avg_acc = 0
#     for i in range(0, train_inputs.shape[0], model.batch_size):
#         batch = train_inputs[i:i + model.batch_size]
#         label_batch = train_labels[i:i + model.batch_size]
#         # Implement backprop:
#         with tf.GradientTape() as tape:
#             probs = model.call(batch) # this calls the call function conveniently
#             loss = model.loss(probs, label_batch)
#             acc = model.accuracy(probs, label_batch)
#             avg_loss += loss
#             avg_acc += acc

#         # The keras Model class has the computed property trainable_variables to conveniently
#         # return all the trainable variables you'd want to adjust based on the gradients
#         gradients = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#         num_batches += 1

#     avg_loss /= num_batches
#     avg_acc /= num_batches
#     return avg_loss, avg_acc

# # def train2(model, train_inputs, train_labels):


# def test(model, test_inputs, test_labels):
#     # fix array indices error
#     """
#     Runs through one epoch - all testing examples

#     :param model: the trained model to use for prediction
#     :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
#     :param test_labels: train labels (all labels for testing) of shape (num_labels,)
#     :returns: perplexity of the test set
#     """
#     #NOTE: Ensure a correct perplexity formula (different from raw loss)
#     avg_loss = 0.0
#     num_batches = 0
#     for i in range(0, test_inputs.shape[0], model.batch_size):
#         num_batches += 1
#         batch = test_inputs[i:i + model.batch_size]
#         label_batch = test_labels[i:i + model.batch_size]
#         probs = model.call(batch)
#         # Get the average loss for each row in probs (2d size batch_size x vocab_size)
#         # Add to overall average loss for each batch
#         avg_loss += model.loss(probs, label_batch)
    
#     avg_loss = avg_loss / num_batches
#     return avg_loss
