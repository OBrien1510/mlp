import numpy as np
from MultiLayerPerceptron import MultiLayerPerceptron
import pandas as pd
from matplotlib import pyplot as plt

output_file = "/home/hugh/connect_comp/ProgrammingAssignment/Task3.txt"

df = pd.read_csv("/home/hugh/connect_comp/letter-recognition.data", header=None)

y = df[0]
x = df.loc[:, df.columns != 0]
# normalize dataset
x = (x-x.min())/(x.max()-x.min())

# split into train test split
index = np.random.rand(len(y)) < 0.9

x_train = np.array(x[index])
y_train = np.array(y[index])
x_test = np.array(x[~index])
y_test = np.array(y[~index])

mlp = MultiLayerPerceptron((len(x.columns), 20, len(y.unique()),), max_iters=200, hidden_activation="relu", output_activation="softmax",
                           linear_factor=1, learning_rate=0.01, verbose=(True, 100), weight_update=1, loss="crossentropy")

errors = mlp.fit(x_train, y_train)

plt.plot(errors, color="blue", label="Relu + Softmax")
plt.title("Cross Entropy Error Over Time")
plt.xlabel("$Epochs$")
plt.ylabel("$Error$")

mlp = MultiLayerPerceptron((len(x.columns), 20, len(y.unique()),), max_iters=200, hidden_activation="Sigmoid", output_activation="softmax",
                           linear_factor=1, learning_rate=0.01, verbose=(True, 100), weight_update=1, loss="crossentropy")

errors = mlp.fit(x_train, y_train)

plt.plot(errors, color="red", label="Sigmoid + Softmax")

mlp = MultiLayerPerceptron((len(x.columns), 20, len(y.unique()),), max_iters=200, hidden_activation="tanh", output_activation="softmax",
                           linear_factor=1, learning_rate=0.01, verbose=(True, 100), weight_update=1, loss="crossentropy")

errors = mlp.fit(x_train, y_train)

plt.plot(errors, color="orange", label="Tanh + Softmax")

mlp = MultiLayerPerceptron((len(x.columns), 20, len(y.unique()),), max_iters=200, hidden_activation="relu", output_activation="linear",
                           linear_factor=1, learning_rate=0.01, verbose=(True, 100), weight_update=1, loss="crossentropy")

errors = mlp.fit(x_train, y_train)

plt.plot(errors, color="magenta", label="Relu + Linear")

mlp = MultiLayerPerceptron((len(x.columns), 20, len(y.unique()),), max_iters=200, hidden_activation="Sigmoid", output_activation="linear",
                           linear_factor=1, learning_rate=0.01, verbose=(True, 100), weight_update=1, loss="crossentropy")

errors = mlp.fit(x_train, y_train)

plt.plot(errors, color="cyan", label="Sigmoid + Linear")

mlp = MultiLayerPerceptron((len(x.columns), 20, len(y.unique()),), max_iters=200, hidden_activation="tanh", output_activation="linear",
                           linear_factor=1, learning_rate=0.01, verbose=(True, 100), weight_update=1, loss="crossentropy")

errors = mlp.fit(x_train, y_train)

plt.plot(errors, color="green", label="Tanh + Linear")

plt.legend()
#plt.show()
plt.savefig("/home/hugh/connect_comp/ProgrammingAssignment/Task3a.png")

correct = 0
total = 0
#file = open("/home/hugh/connect_comp/ProgrammingAssignment/Task3.txt", 'a+')
#file.write("\n\n")
#file.close()
for i in range(len(y_test)):

    with open("/home/hugh/connect_comp/ProgrammingAssignment/Task3.txt", 'a+') as fh:
        prediction = mlp.predict(x_test[i])
        #fh.write("Prediction {0}\n".format(prediction))
        #fh.write("Actual {0}\n".format(y_test[i]))
        if prediction == y_test[i]:
            correct += 1

        total += 1

#with open("/home/hugh/connect_comp/ProgrammingAssignment/Task3.txt", 'a+') as fh:
    #fh.write("Percentage accuracy on test set: {0}".format((correct/total)*100))

#mlp.to_str()

