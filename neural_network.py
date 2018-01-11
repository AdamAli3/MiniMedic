import numpy as np
import pandas as pd
import math

class NeuralNetwork():

    def __init__(self, input_count, seed):
        np.random.seed(seed)
        self.normal_contstants = []
        self.input_count = input_count
        self.generate_random_weights()

    def softmax(self, x):
        expox = [math.exp(i) for i in x]
        sum_expox = sum(expox)
        final = [round(i / sum_expox, 3) for i in expox]
        return final

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train_predict(self, given_input):
        scores = []
        for index, row in given_input.iterrows():
            scores.append(sum(self.synaptic_weights * row))
        scores = self.softmax(scores)
        outputs = []
        for i in scores:
            outputs.append(self.sigmoid(i))
        return outputs

    def generate_random_weights(self):
        self.synaptic_weights = np.random.uniform(0, 1, size=(self.input_count))
        self.bias = np.random.uniform(0, 1, 1)

    def normalize_training_input(self, training_input):
        for column in training_input:
            self.normal_contstants.append((training_input[column].max() - training_input[column].min()) / 2)

        for i in range(self.input_count):
            distance = self.normal_contstants[i]
            training_input.iloc[:, i] = (training_input.iloc[:, i].transform(lambda x: (x - distance) / distance))

    def train(self, training_input, training_output, iterations):
        self.normalize_training_input(training_input)
        print("Synaptic Weights: ", self.synaptic_weights)
        print("Bias: ", self.bias)
        output = (self.train_predict(training_input))
        error = (training_output - output)
        print(error)
        print(training_input)
        print(self.normal_contstants)
