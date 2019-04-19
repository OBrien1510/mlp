import numpy as np
from MultiLayerPerceptron import MultiLayerPerceptron

input_vector = list()
output_vector = list()

def combine(x):

    return np.sin(x[0]-x[1]+x[2]+x[3])

for i in range(200):

    unit = (np.random.random(4) - 0.5) * 2
    input_vector.append(unit)
    output_vector.append(combine(unit))


input_vector = np.array(input_vector)
output_vector = np.array(output_vector)

index = np.random.rand(len(input_vector)) < 0.8

x_train = input_vector[index]
y_train = output_vector[index]
x_test = input_vector[~index]
y_test = output_vector[~index]

mlp = MultiLayerPerceptron((4,10,1), hidden_activation="sigmoid", max_iters=5000, linear_factor=0.05, learning_rate=0.015,
                           verbose=(True, 100))

mlp.fit(x_train, y_train)

errors = list()

for i in range(len(x_test)):
    prediction = mlp.predict(x_test[i])
    error = (prediction-y_test[i])
    errors.append(error)

print("average test error", np.mean(np.absolute(errors)))
print()
