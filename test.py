from MultiLayerPerceptron import MultiLayerPerceptron
import numpy as np

mlp = MultiLayerPerceptron((2,2,1))

starting_weights = mlp.weights

x = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
y = np.array([1, 1, 0, 0])

mlp.fit(x, y)


print("1, 1")
print(mlp.predict(np.array([1, 1])))
print("0, 0")
print(mlp.predict(np.array([0, 0])))
print("0, 1")
print(mlp.predict(np.array([0, 1])))
print("1, 0")
print(mlp.predict(np.array([1, 0])))

#print(mlp.raw_neuron_inputs)

final_weights = mlp.weights
#print("weights @ start", starting_weights)
print("weights @ end", final_weights)
print("biases @ end", mlp.layer_biases)

#print("inputs", mlp.neuron_inputs)

#print("outputs", mlp.neuron_outputs)

#print("errors", mlp.neuron_errors)

mlp.to_str()

