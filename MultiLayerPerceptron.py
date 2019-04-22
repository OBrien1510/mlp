import numpy as np
import random
from scipy.special import softmax

class MultiLayerPerceptron:

    def __init__(self, layers=(2, 2, 1, ), hidden_activation="sigmoid", output_activation="linear", linear_factor=0.05, max_iters=20000, learning_rate=0.5, verbose=(True, 10,), weight_update=10, loss="mse", train=True, output_file=None):

        self.layers = layers
        self.output_file = output_file
        self.train = train
        self.no_input = layers[0]
        self.no_output = layers[len(layers)-1]
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.hidden_activation_f, self.hidden_d_activation_f = self.get_activiation(hidden_activation)
        self.output_activation_f, self.output_d_activation_f = self.get_activiation(output_activation)
        self.linear_factor = linear_factor
        self.error_f = self.get_error(loss)
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        self.weight_update = weight_update
        self.verbose = verbose[0]
        self.notification = verbose[1]
        self.weights = list()
        self.weight_changes = list()
        self.neuron_inputs = list()
        self.neuron_outputs = list()
        self.raw_neuron_inputs = list()
        self.bias_changes = list()
        self.layer_biases = list()
        self.errors = list()

        # initialize all required arrays
        for i, layer in enumerate(layers[1:]):
            self.layer_biases.append(np.random.uniform(-0.1, 0.1, (layer,)))
            self.bias_changes.append(np.random.uniform(-0.1, 0.1, (layer,)))
            self.weights.append(np.random.uniform(-0.1, 0.1, (layers[i], layer)))
            self.weight_changes.append(np.zeros((layers[i], layer)))
            self.neuron_inputs.append(list())
            self.raw_neuron_inputs.append(list())
            self.neuron_outputs.append(list())


    def get_error(self, error):

        if error == "mse":
            return lambda x, y: 0.5*np.power((x-y), 2)
        else:
            return lambda x, y: -1 * np.sum(y * (x + (-x.max() - np.log(np.sum(np.exp(x-x.max()))))))

    def get_activiation(self, activation):

        if activation.lower() == "sigmoid":

            return lambda x: 1/(1+np.exp(-x)), lambda x: x*(1-x)

        if activation.lower() == "linear":

            return lambda x: self.linear_factor*x, lambda x: self.linear_factor

        if activation.lower() == "relu":

            return lambda x: np.maximum(0, x), lambda x: np.maximum(0, x/np.absolute(x)) if x.any() > 0 else 0

        if activation.lower() == "tanh":
            return lambda x: np.sinh(x)/np.cosh(x), lambda x: 1 - (np.tanh(x)**2)

        else:
            return (None, None)

    def fit(self,x, y):

        x = np.array(x)
        y = np.array(y)

        # if more than one output layer and training output hasn't been preprocessed, prepare y for classification
        if y.shape[0] != len(np.unique(y)) and self.layers[-1] > 1:

            y = self.prep_class(y)

        return self.gradient_descent(x, y)


    def softmax(self, x):

        return softmax(x)

    def relu(self, x):

       return np.where(x > 0, x, x*0.01)

    def relu_d(self,x ):

        return np.where(x > 0, x, 0.01)

    def prep_class(self, y):

        self.unique = list(np.unique(y))

        final_y = list()

        for i, value in enumerate(y):

            new_y = np.zeros(len(self.unique))
            index = self.unique.index(value)
            new_y[index] = 1
            final_y.append(new_y)

        return np.array(final_y)

    def gradient_descent(self, x, y):

        iterations = 0
        epoch_errors = list()
        while iterations < self.max_iters:

            errors = list()
            for i in range(len(y)):

                prediction = self.predict(x[i], train=True)

                if self.output_activation != "softmax":
                    error = self.error_f(prediction, y[i])
                    backprop = (prediction-y[i])
                else:
                    error = self.error_f(prediction, y[i])
                    # derivative of softmax is prediction - target
                    # where target is a vector of hot encoding results
                    backprop = prediction - y[i]
                errors.append(np.mean(error))
                self.backpropagate(error, backprop, 0)
                # update after every 2 examples
                if i % 2 == 0:
                    self.update_weights()

            if iterations % self.notification == 0 and self.verbose:

                #randint = random.randint(0, len(x)-1)
                print("Epoch #", iterations)
                #print("Random Sample:", (y[randint]))
                #print("Prediction of Sample", self.predict(x[randint]))
                mean_error = np.mean(errors)
                print("Average error", mean_error)
                if self.output_file is not None:
                    with open(self.output_file, 'a+') as fh:
                        fh.write("Epoch #{0}\n".format(iterations))
                        mean_error = np.mean(errors)
                        fh.write("Average error: {0:.16f}\n".format(mean_error))

            epoch_errors.append(np.mean(errors))
            iterations += 1

        return epoch_errors


    def reset(self):

        self.neuron_inputs = list()
        self.neuron_outputs = list()
        self.raw_neuron_inputs = list()

        for i, layer in enumerate(self.layers[1:]):

            self.neuron_inputs.append(list())
            self.raw_neuron_inputs.append(list())
            self.neuron_outputs.append(list())


    def backpropagate(self, delta, total_error, depth):

        if depth == len(self.weights):

            return

        if depth == 0:

            # treat the output layer differently from hidden layers
            output = self.neuron_outputs[-1]
            if self.output_activation != "softmax":
                derivative = self.output_d_activation_f(np.array(output[0]))

                delta = total_error*derivative

            else:

                delta = total_error

            # multiply delta by output of previous hidden layer
            output = self.neuron_outputs[-2][0].reshape(self.neuron_outputs[-2][0].shape[0], 1)
            x = np.dot(output, delta.reshape(1, delta.shape[0]))

            # get error for each hidden node for back propogation
            y = np.dot(self.weights[-1], delta)

            self.weight_changes[-1] += x
            self.bias_changes[-1] += delta

            # pass errors back to next layer
            self.backpropagate(y, total_error, depth+1)

        else:

            index = len(self.weights) - depth - 1

            # deal with hidden layers
            current_n_layer = self.neuron_outputs[index][0]
            current_in_layer = np.squeeze(self.raw_neuron_inputs[index][0][0])
            if self.hidden_activation != "relu":
                current_derivative = self.hidden_d_activation_f(current_n_layer)
            else:
                current_derivative = self.relu_d(current_n_layer)

            x = current_derivative*delta

            y = np.dot(current_in_layer.reshape(current_in_layer.shape[0], 1), x.reshape(1, len(delta)))

            self.bias_changes[index] += x
            self.weight_changes[index] += y

            # pass layer's error to the next layer
            y = np.dot(self.weights[index], x)
            self.backpropagate(y, total_error, depth+1)


    def update_weights(self):

        for i in range(len(self.weights)):

            self.weights[i] = np.subtract(self.weights[i], self.learning_rate * self.weight_changes[i])
            self.layer_biases[i] = np.subtract(self.layer_biases[i], self.learning_rate * self.bias_changes[i])
            self.weight_changes[i] = np.zeros(self.weight_changes[i].shape)
            self.bias_changes[i] = np.zeros(self.bias_changes[i].shape)

    def predict(self, x, train=False):

        input = x

        activation_outputs = 0

        self.reset()

        for i, layer in enumerate(self.weights):

            input = input.reshape(input.shape[0], 1)
            self.raw_neuron_inputs[i].append(np.repeat(np.array([input]), [self.layers[i+1]], axis=0))
            activation_inputs = layer.T.dot(input)

            self.neuron_inputs[i].append(activation_inputs)

            # if layer is a a hidden layer
            if i != len(self.weights)-1:

                if self.hidden_activation != "relu":
                    activation_outputs = self.hidden_activation_f(activation_inputs.T[0] + self.layer_biases[i])
                else:
                    activation_outputs = self.relu(activation_inputs.T[0] + self.layer_biases[i])

            else:

                activation_inputs = activation_inputs.reshape(len(activation_inputs))

                if self.output_activation != "softmax":

                    activation_outputs = self.output_activation_f(activation_inputs + self.layer_biases[i])
                else:

                    activation_outputs = self.softmax(activation_inputs + self.layer_biases[i])

            self.neuron_outputs[i].append(activation_outputs)

            input = activation_outputs

        if train:

            return activation_outputs
        else:
            if self.layers[-1] > 1:

                prediction = np.max(activation_outputs)
                return self.unique[list(activation_outputs).index(prediction)]
            else:
                return activation_outputs

    def to_str(self):

        print("Input Layer Input")
        print("------------------")
        print(self.raw_neuron_inputs[0])

        for i, layer in enumerate(self.layers[1:]):
            print("Layer #",i+2)
            print("Weights")
            print("--------------------")
            print(self.weights[i])
            print("Layer Biases")
            print("---------------------")
            print(self.layer_biases[i])
            print("Outputs")
            print("---------------------")
            print(self.neuron_outputs[i])
            print("---------------------")





