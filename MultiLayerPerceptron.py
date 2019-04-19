import numpy as np
import random
from scipy.special import softmax

class MultiLayerPerceptron:

    def __init__(self, layers=(2, 2, 1, ), hidden_activation="sigmoid", output_activation="linear", linear_factor=0.05, max_iters=20000, learning_rate=0.5, verbose=(True, 10,), weight_update=10, loss="mse", train=True):

        self.layers = layers
        self.train = train
        self.no_input = layers[0]
        self.no_output = layers[len(layers)-1]
        self.activation = output_activation
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
        np.random.seed(3)

        # initialize all required arrays
        for i, layer in enumerate(layers[1:]):
            self.layer_biases.append(np.random.uniform(-0.1, 0.1, (layer,)))
            self.bias_changes.append(np.random.uniform(-0.1, 0.1, (layer,)))
            self.weights.append(np.random.uniform(-0.1, 0.1, (layer, layers[i])))
            self.weight_changes.append(np.zeros((layer, layers[i])))
            self.neuron_inputs.append(list())
            self.raw_neuron_inputs.append(list())
            self.neuron_outputs.append(list())

    def get_error(self, error):

        if error == "mse":
            return lambda x, y: 0.5*np.power((x-y), 2)
        else:
            return lambda x, y: -(np.sum(y*np.log(x)))/len(x)

    def get_activiation(self, activation):

        if activation.lower() == "sigmoid":

            return lambda x: 1/(1+np.exp(-x)), lambda x: x*(1-x)

        if activation.lower() == "linear":

            return lambda x: self.linear_factor*x, lambda x: self.linear_factor

        if activation.lower() == "relu":

            return lambda x: np.maximum(0, x), lambda x: np.maximum(0, x/np.absolute(x)) if x.any() > 0 else 0

        else:
            return (None, None)

    def fit(self,x, y):

        x = np.array(x)
        y = np.array(y)

        # if more than one output layer and training output hasn't been preprocessed, prepare y for classification
        if y.shape[0] != len(np.unique(y)) and self.layers[-1] > 1:

            y = self.prep_class(y)

        self.gradient_descent(x, y)


    def softmax(self, x):

        return softmax(x)

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

        error = 1
        iterations = 1
        prediction = 0
        i = 0
        mean_error = 1
        change_error = 1

        while iterations <= self.max_iters:
            #print(iterations)
            errors = list()
            for i in range(len(y)):

                prediction = self.predict(x[i], train=True)

                if self.activation != "softmax":
                    error = self.error_f(prediction, y[i])
                    backprop = (prediction-y[i])
                else:
                    error = self.error_f(prediction, y[i])
                    # derivative of softmax is prediction - target
                    # where target is a vector of hot encoding results
                    backprop = prediction - y[i]
                errors.append(np.mean(error))
                self.backpropagate(error, backprop, 0)

            if iterations % self.weight_update == 0:
                self.update_weights()

            if iterations % self.notification == 0 and self.verbose:

                randint = random.randint(0, len(x)-1)
                print("Epoch #", iterations)
                print("Random Sample:", y[randint])
                print("Prediction of Sample", self.predict(x[randint], train=True))
                change_error = abs(np.mean(error) - np.mean(errors))/np.mean(error)
                mean_error = np.mean(errors)
                print("Average error", mean_error)
                #self.errors.append(mean_error)



            iterations += 1


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
            if self.activation != "softmax":
                derivative = self.output_d_activation_f(np.array(output[0]))

                delta = total_error*derivative

            else:

                delta = total_error

            new_delta = list()

            # multiply delta by output of previous hidden layer

            # this following for loop is messy as a result of me trying to strong arm
            # my algorithm to work with classification when it wasn't initially build with
            # classification in mind

            for i, output in enumerate(self.neuron_outputs[-2][0]):

                # weights are arranged by layer instead of by neuron, however for
                # this case we want then to be arranged by neuron so reshape
                weight_shape = self.weight_changes[-1].shape
                print("original", self.weight_changes[-1][0])
                temp_weight = self.weight_changes[-1].reshape(weight_shape[1], weight_shape[0])
                print("temp", temp_weight[0])
                temp_weight += output * delta
                # reshape back to original for future compatability
                temp_weight = temp_weight.reshape(weight_shape[0], weight_shape[1])

                self.weight_changes[-1] = temp_weight
                self.bias_changes[-1] += delta

            weight_shape = self.weight_changes[-1].shape

            for weight in self.weights[-1].reshape(weight_shape[1], weight_shape[0]):

                new_delta.append(delta*weight)

            # pass errors back to next layer according to current layers weights
            self.backpropagate(np.array(new_delta), total_error, depth+1)

        else:

            index = len(self.weights) - depth - 1

            # deal with hidden layers
            current_n_layer = self.neuron_outputs[index][0]
            current_in_layer = self.raw_neuron_inputs[index][0]
            current_derivative = self.hidden_d_activation_f(current_n_layer)

            # weights of current node * current node input * derivative of activation wrt to current node's output
            # each element of current_w_layer represents a (2,) array
            # the other two arrays are 1 dimensional
            wxin = np.array([value * current_derivative[i] for i, value in enumerate(current_in_layer)])

            wxin_b = np.array([value for i, value in enumerate(current_derivative)])

            # multiply each value by the error from the previous layer
            error_transfer = np.sum(delta) * wxin
            bias_change = np.sum(delta) * wxin_b

            self.bias_changes[index] += bias_change
            self.weight_changes[index] += error_transfer

            # pass layer's error to the next layer
            self.backpropagate(sum(error_transfer), total_error, depth+1)


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

            #print("layer", layer)
            self.raw_neuron_inputs[i].append(np.repeat(np.array([input]), [len(layer)], axis=0))

            activation_inputs = layer.dot(input)

            self.neuron_inputs[i].append(activation_inputs)

            # if layer is a a hidden layer
            if i != len(self.weights)-1:

                activation_outputs = self.hidden_activation_f(activation_inputs + self.layer_biases[i])


            else:

                if self.activation != "softmax":
                    activation_outputs = self.output_activation_f(activation_inputs + self.layer_biases[i])
                else:
                    activation_outputs = self.softmax(activation_inputs + self.layer_biases[i])

            #print("outputs", activation_outputs)
            self.neuron_outputs[i].append(activation_outputs)

            input = activation_outputs

        if train:
            return activation_outputs

        else:
            if self.layers[-1] > 1:
                #print(activation_outputs)
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





