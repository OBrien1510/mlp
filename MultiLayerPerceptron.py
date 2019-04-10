import numpy as np

class MultiLayerPerceptron:

    def __init__(self, layers=(2, 2, 1, ), activation="sigmoid", max_iters=50000, learning_rate=0.5):

        self.layers = layers
        self.no_input = layers[0]
        self.no_output = layers[len(layers)-1]
        self.activation_f, self.d_activation_f = self.get_activiation(activation)
        self.error_f = lambda x, y: 0.5*np.power((x-y), 2)
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        self.weights = list()
        self.weight_changes = list()
        self.neuron_inputs = list()
        self.neuron_outputs = list()
        self.raw_neuron_inputs = list()
        self.bias_changes = list()
        self.layer_biases = list()
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

        print("biases @ start", self.layer_biases)
        print("weights @ start", self.weights)

    def get_activiation(self, activation):

        if activation.lower() == "sigmoid":

            return lambda x: 1/(1+np.exp(-x)), lambda x: x*(1-x)

    def fit(self,x, y):

        self.gradient_descent(x, y)



    def gradient_descent(self, x, y):

        error = 1
        iterations = 1
        prediction = 0
        i = 0

        while iterations <= self.max_iters:

            errors = np.ones(shape=(len(y),))
            for i in range(len(y)):

                prediction = self.predict(x[i])
                #print("x", x[i])
                #print("y", y[i])
                #print("prediction", prediction)
                error = self.error_f(prediction, y[i])
                #print("error", error)


                errors[i] = error
                self.backpropagate(error, (prediction-y[i]), 0)

            if iterations % 10 == 0:
                self.update_weights()

            if iterations % 100 == 0:
                print("Epoch #", iterations)
                print("Average error", np.mean(errors))
                print("prediction of 1, 0:",self.predict(np.array([1, 0])))


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
            derivative = self.d_activation_f(np.array(output[0]))
            delta = total_error*derivative

            # divide current inputs by the weights to the previous layer to get the outputs of the previous layer
            inputs = np.divide(self.neuron_inputs[-1][0], self.weights[-1])
            self.weight_changes[-1] += delta*self.neuron_outputs[-2]
            self.bias_changes[-1] += delta

            # pass errors back to next layer according to current layers weights
            self.backpropagate(delta*self.weights[-1], total_error, depth+1)

        else:

            index = len(self.weights) - depth - 1

            # deal with hidden layers
            current_w_layer = self.weights[index]
            current_n_layer = self.neuron_outputs[index][0]
            current_in_layer = self.raw_neuron_inputs[index][0]
            current_derivative = self.d_activation_f(current_n_layer)

            # weights of current node * current node input * derivative of activation wrt to current node's output
            # each element of current_w_layer represents a (2,) array
            # the other two arrays are 1 dimensional
            wxin = np.array([value * current_derivative[i] for i, value in enumerate(current_in_layer)])

            wxin_b = np.array([value for i, value in enumerate(current_derivative)])

            #wxin_b = np.array([sum(i) for i in wxin_b])
            #print("wxin_b", wxin_b)

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

    def predict(self, x):

        input = x

        activation_outputs = 0

        self.reset()

        for i, layer in enumerate(self.weights):

            self.raw_neuron_inputs[i].append(np.repeat(np.array([input]), [len(layer)], axis=0))

            activation_inputs = layer.dot(input)

            self.neuron_inputs[i].append(activation_inputs)

            #print(activation_inputs)
            #print(self.layer_biases[i])

            activation_outputs = self.activation_f(activation_inputs + self.layer_biases[i])

            self.neuron_outputs[i].append(activation_outputs)

            input = activation_outputs


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





