import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import random

iris = datasets.load_iris()
iris_data = iris.data
sl_data = iris_data[:100, 0]
sw_data = iris_data[:100, 1]

sl_ave = np.average(sl_data)
sl_data -= sl_ave
sw_ave = np.average(sw_data)
sw_data -= sw_ave

train_data = []
for i in range(100):
    correct = iris.target[i]
    correct = iris.target[i]
    train_data.append([sl_data[i], sw_data[i], correct])


# シグモイド関数
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ニューロン
class Neuron:
    def __init__(self):
        self_input = 0.0
        self.output = 0.0

    def set_input(self, inp):
        self.input_sum += inp

    def get_output(self):
        self.output = sigmoid(self.input_sum)
        return self.output

    def reset(self):
        self.input_sum = 0
        self.output = 0


class NeuralNetwork:
    def __init__(self):
        self.w_im = [[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]]
        self.w_mo = [[1.0, -1.0, 1.0]]

        self.b_m = [3.0, 0.0, -3.0]
        self.b_o = [-0.5]

        self.input_layer = [0.0, 0.0]
        self.middle_layer = [Neuron(), Neuron(), Neuron()]
        self.output_layer = [Neuron()]

    def commit(self, input_data):
        # 各層のリセット
        self.input_layer[0] = input_data[0]  # 入力層は値を受け取るのみ
        self.input_layer[1] = input_data[1]
        self.middle_layer[0].reset()
        self.middle_layer[1].reset()
        self.middle_layer[2].reset()
        self.output_layer[0].reset()

        self.middle_layer[0].set_input(self.input_layer[0] * self.w_im[0][0])
        self.middle_layer[0].set_input(self.input_layer[1] * self.w_im[0][1])
        self.middle_layer[0].set_input(self.b_m[0])

        self.middle_layer[1].set_input(self.input_layer[0] * self.w_im[1][0])
        self.middle_layer[1].set_input(self.input_layer[1] * self.w_im[1][1])
        self.middle_layer[1].set_input(self.b_m[1])

        self.middle_layer[2].set_input(self.input_layer[0] * self.w_im[2][0])
        self.middle_layer[2].set_input(self.input_layer[1] * self.w_im[2][1])
        self.middle_layer[2].set_input(self.b_m[2])

        # 中間層→出力層
        self.output_layer[0].set_input(self.middle_layer[0].get_output() * self.w_mo[0][0])
        self.output_layer[0].set_input(self.middle_layer[1].get_output() * self.w_mo[0][1])
        self.output_layer[0].set_input(self.middle_layer[2].get_output() * self.w_mo[0][2])
        self.output_layer[0].set_input(self.b_o[0])

        return self.output_layer[0].get_output()

    def train(self, correct):
        # Learning Coefficient
        k = 0.3

        # output
        output_o = self.output_layer[0].output
        output_m0 = self.middle_layer[0].output
        output_m1 = self.middle_layer[1].output
        output_m2 = self.middle_layer[2].output

        # δ
        delta_o = (output_o - correct) * output_o * (1.0 - output_o)
        delta_m0 = delta_o * self.w_mo[0][0] * output_m0 * (1 - output_m0)
        delta_m1 = delta_o * self.w_mo[0][1] * output_m1 * (1 - output_m1)
        delta_m2 = delta_o * self.w_mo[0][2] * output_m2 * (1 - output_m2)

        # update parameters
        self.w_mo[0][0] -= k * delta_o * output_m0
        self.w_mo[0][1] -= k * delta_o * output_m1
        self.w_mo[0][2] -= k * delta_o * output_m2
        self.b_o[0] -= k * delta_o

        #  self.w_im[0][0] -= k * delta_m0 * self.input_layer[0]
        self.w_im[0][0] -= k * delta_m0 * self.input_layer[0]
        self.w_im[0][1] -= k * delta_m0 * self.input_layer[1]
        self.w_im[1][0] -= k * delta_m1 * self.input_layer[0]
        self.w_im[1][1] -= k * delta_m1 * self.input_layer[1]
        self.w_im[2][0] -= k * delta_m2 * self.input_layer[0]
        self.w_im[2][1] -= k * delta_m2 * self.input_layer[1]
        self.b_m[0] -= k * delta_m0
        self.b_m[1] -= k * delta_m1
        self.b_m[2] -= k * delta_m2


# Instance of Neuralnetwork
neural_network = NeuralNetwork()


def show_graph(epoch):
    print("Epoch:", epoch)
    # Execute
    st_predicted = [[], []]  # Setosa
    vc_predicted = [[], []]  # Versicolor
    for data in train_data:
        if neural_network.commit(data) < 0.5:
            st_predicted[0].append(data[0] + sl_ave)
            st_predicted[1].append(data[1] + sw_ave)
        else:
            vc_predicted[0].append(data[0] + sl_ave)
            vc_predicted[1].append(data[1] + sw_ave)

    # Show graph of categorized result
    plt.scatter(st_predicted[0], st_predicted[1], label="Setosa")
    plt.scatter(vc_predicted[0], vc_predicted[1], label="Versicolor")
    plt.legend()

    plt.xlabel("Sepal length (cm)")
    plt.ylabel("Sepal width (cm)")
    plt.show()


show_graph(0)

# Show result after training
for t in range(0, 64):
    random.shuffle(train_data)
    for data in train_data:
        neural_network.commit(data[:2])  # forward propagation
        neural_network.train(data[2])  # backward propagation
    if t + 1 in [1, 2, 4, 8, 16, 32, 64]:
        show_graph(t + 1)

# Show original graph to compare
st_data = iris_data[:50]
vc_data = iris_data[50:100]
plt.scatter(st_data[:, 0], st_data[:, 1], label="Setosa")
plt.scatter(vc_data[:, 0], vc_data[:, 1], label="Versicolor")
plt.legend()

plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.title("original")
plt.show()