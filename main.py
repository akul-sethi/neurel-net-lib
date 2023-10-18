from random import random
from CSVProcessor import convert_training, print_picture, convert_testing

import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


matrix_sig = np.vectorize(sigmoid)


def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


matrix_dsig = np.vectorize(dsigmoid)


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.ih_weights = np.random.randn(input_nodes, hidden_nodes)
        self.ho_weights = np.random.randn(hidden_nodes, output_nodes)

        self.h_biases = np.random.randn(hidden_nodes, 1)
        self.o_biases = np.random.randn(output_nodes, 1)

        self.loss = 0

    # Input_vector should be an array
    def run(self, input_array):
        input_vector = np.array([input_array]).transpose()
        if input_vector.size != self.input_nodes:
            print("Incorrect input vector size")

        ih_weights_transposed = np.transpose(self.ih_weights)
        ho_weights_transposed = np.transpose(self.ho_weights)

        # Here I am using 3blue1brown convention
        hidden_z = np.add(np.dot(ih_weights_transposed, input_vector), self.h_biases)
        hidden_a = matrix_sig(hidden_z)

        output_z = np.add((ho_weights_transposed @ hidden_a), self.o_biases)
        output_a = matrix_sig(output_z)

        return output_a

    # training data should be array of input output tuples(each input and output should be an array)
    def train(self, training_data, cycles):
        step = 0.5
        for cycle in range(cycles):
            random_sample = math.floor(random() * len(training_data))
            training_input_array = training_data[random_sample][0]
            training_output_array = training_data[random_sample][1]

            training_input = np.array([training_input_array]).transpose()
            training_output = np.array([training_output_array]).transpose()

            ih_weights_transposed = np.transpose(self.ih_weights)
            ho_weights_transposed = np.transpose(self.ho_weights)

            # Here I am using 3blue1brown convention
            hidden_z = np.add((ih_weights_transposed @ training_input), self.h_biases)
            hidden_a = matrix_sig(hidden_z)

            output_z = np.add((ho_weights_transposed @ hidden_a), self.o_biases)
            actual_output = matrix_sig(output_z)

            # Now that everything has been forward propagated, we need to calculate the partial derivatives
            error = actual_output - training_output

            o_baises_gradient = error * matrix_dsig(output_z)

            hidden_a_expansion = np.repeat(hidden_a, self.output_nodes, 1)
            cost_in_terms_z = (error * matrix_dsig(output_z)).transpose()
            cost_in_terms_z_expansion = np.repeat(cost_in_terms_z, self.hidden_nodes, 0)
            ho_weights_gradient = cost_in_terms_z_expansion * hidden_a_expansion

            hidden_partial_derivative = self.ho_weights @ (error * matrix_dsig(output_z))

            h_baises_gradient = hidden_partial_derivative * matrix_dsig(hidden_z)

            training_input_expansion = np.repeat(training_input, self.hidden_nodes, 1)
            hidden_cost_in_terms_z = (hidden_partial_derivative * matrix_dsig(hidden_z)).transpose()
            hidden_cost_in_terms_z_expansion = np.repeat(hidden_cost_in_terms_z, self.input_nodes, 0)
            ih_weights_gradient = hidden_cost_in_terms_z_expansion * training_input_expansion

            # Change the weights and biases
            self.h_biases = self.h_biases - (step * h_baises_gradient)
            self.o_biases = self.o_biases - (step * o_baises_gradient)

            self.ih_weights = self.ih_weights - (step * ih_weights_gradient)
            self.ho_weights = self.ho_weights - (step * ho_weights_gradient)

    def run_and_choose(self, input):
        output_array = nn.run(input)
        best_result = 0
        for number in range(len(output_array)):
            if output_array[number] > output_array[best_result]:
                best_result = number

        return best_result

    def prc(self):
        random_picture = math.floor(random() * 1000)
        print_picture(testing_array[random_picture])
        return self.run_and_choose(testing_array[random_picture])

    def percent_correct(self, ta):
        amount_correct = 0
        total_samples = 0
        for sample in ta:
            for number in range(len(sample[1])):
                if sample[1][number] == 1:
                    correct_num = number
            if self.run_and_choose(sample[0]) == correct_num:
                amount_correct = amount_correct + 1
            total_samples = total_samples + 1
        return amount_correct / total_samples

    def loss(self, ta):
        total_loss = 0
        for sample in ta:
            output = self.run(sample[0])
            target = np.array(sample[1]).reshape(len(sample[1]), 1)
            error_vector = output - target
            loss = np.dot(error_vector.transpose(), error_vector)
            total_loss += loss
        return total_loss.item()


sample_training_set = [([1, 1], [0]), ([1, 0], [1]), ([0, 1], [1]), ([0, 0], [0])]
training_array = convert_training('train.csv', 1000)
testing_array = convert_testing('test.csv')
print("Loading data:  completed")

nn = NeuralNetwork(784, 50, 10)
nn.train(training_array, 5000)
print("Training:  completed")
print("Percent Correct: ")
print(nn.percent_correct(training_array))

# print("Loss:  ")
# print(str(nn.loss(training_array)))
