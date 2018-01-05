import numpy as np

class NeuralNetwork():

    def __init__(self, input_count, output_count, seed, labels):
        np.random.seed(seed)
        self.input_count = input_count
        self.output_count = output_count
        self.labels = labels
        self.generate_random_weights()

    def __sigmoid(self, x):
        return (self.labels - 1) * (1 / (1 + np.exp(-x * 2)))

    def __sigmoid_derivative(self, x):
        return (self.labels - 1) * x * (1 - x)

    def generate_random_weights(self):
        self.synaptic_weights = np.random.uniform(0, (self.labels - 1), size=(self.input_count, self.output_count))

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        errors = []
        weights = []

        for iteration in range(number_of_training_iterations):

            output = self.predict(training_set_inputs)

            error = training_set_outputs - output
            errors.append(abs(np.mean(error)))

            adjustment = np.dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            self.synaptic_weights = self.synaptic_weights + adjustment
            weights.append(self.synaptic_weights.copy())

        return errors, weights

    def train_until_fit(self, training_set_inputs, training_set_outputs, error_delta):

        count = 0;

        last_error_mean = 0

        while(True):
            output = self.predict(training_set_inputs)
            error = training_set_outputs - output
            error_mean = np.mean(error)
            check = (sum(abs(error)))

            if (check < error_delta):
                print("Training is complete!")
                print("Training took {0} iterations to get fit".format(count))
                break
            adjustments = np.dot(training_set_inputs.T, error * sel.__sigmoid_derivative(output))
            self.synaptic_weights += adjustments
            count += 1

        return count

    def untrain(self):
        self.generate_random_weights()

    def predict(self, inputs):
        return self.__sigmoid(np.dot(inputs, self.synaptic_weights))
