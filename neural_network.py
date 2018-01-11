import numpy as np
import pandas as pd

class NeuralNetwork():

    def __init__(self):
        self.normal_contstants = []

    def normalize_training_input(self, training_input):
        for column in training_input:
            self.normal_contstants.append((training_input[column].max() - training_input[column].min())/2)

        for i in range(len(self.normal_contstants)):
            distance = self.normal_contstants[i]
            training_input.iloc[:, i] = (training_input.iloc[:, i].transform(lambda x: (x - distance)/distance))

        print(training_input)
