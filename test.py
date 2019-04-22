from MultiLayerPerceptron import MultiLayerPerceptron
import numpy as np
from matplotlib import pyplot as plt

output_file = "/home/hugh/connect_comp/ProgrammingAssignment/Task1.txt"

mlp = MultiLayerPerceptron((2, 2, 1), hidden_activation="sigmoid", max_iters=10000, linear_factor=0.2, learning_rate=0.7,
                           verbose=(True, 1000,), output_activation="linear", output_file=output_file)


x = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
y = np.array([1, 1, 0, 0])

errors = mlp.fit(x, y)

plt.figure()
plt.plot(errors, color="blue", label="Sigmoid + Linear")
plt.title("Mean Squared Error Over Time")
plt.xlabel("$Epochs$")
plt.ylabel("$Error$")
plt.show()
#plt.savefig("/home/hugh/connect_comp/ProgrammingAssignment/Task1.png")

print("1, 1")
print(mlp.predict(np.array([1, 1])))
print("0, 0")
print(mlp.predict(np.array([0, 0])))
print("0, 1")
print(mlp.predict(np.array([0, 1])))
print("1, 0")
print(mlp.predict(np.array([1, 0])))

#mlp.to_str()

