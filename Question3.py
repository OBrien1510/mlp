import numpy as np
from MultiLayerPerceptron import MultiLayerPerceptron
import pandas as pd

df = pd.read_csv("/home/hugh/connect_comp/letter-recognition.data", header=None)

y = df[0]
x = df.loc[:, df.columns != 0]

index = np.random.rand(len(y)) < 0.8

x_train = np.array(x[index])
y_train = np.array(y[index])
x_test = np.array(x[~index])
y_test = np.array(y[~index])

mlp = MultiLayerPerceptron((len(x.columns), 50, len(y.unique()),), max_iters=10000, hidden_activation="sigmoid", output_activation="softmax",
                           linear_factor=0.01, learning_rate=0.001, verbose=(True, 1), weight_update=5, loss="crossentropy")

mlp.fit(x_train, y_train)
correct = 0
total = 0
for i in range(len(y_test)):

    prediction = mlp.predict(x_test[i])
    print("Prediction", prediction)
    print("Actual", y[i])
    if prediction == y[i]:
        correct += 1

    total += 1

print("Percentage accuracy:", (correct/total)*100)

mlp.to_str()

