import numpy as np
from Neuron import Neuron
from Helpers import deriv_sigmoid


class NeuralNetwork:

    def __init__(self):

        # init hidden layout with 3 neurons.
        self.hidden_layout = np.array([
            Neuron(),
            Neuron(),
            Neuron()
        ])

        # init output layout with 3 neurons.
        self.output_layout = np.array([
            Neuron(),
            Neuron(),
            Neuron()
        ])

    def feedforward(self, inputs):
        # inputs is a numpy array with 3 elements.
        self.hidden_layout[0].calculate_neuron_value(inputs)
        self.hidden_layout[1].calculate_neuron_value(inputs)
        self.hidden_layout[2].calculate_neuron_value(inputs)

        self.output_layout[0].calculate_neuron_value(np.array([
            self.hidden_layout[0].value,
            self.hidden_layout[1].value,
            self.hidden_layout[2].value
        ]))

        self.output_layout[1].calculate_neuron_value(np.array([
            self.hidden_layout[0].value,
            self.hidden_layout[1].value,
            self.hidden_layout[2].value
        ]))

        self.output_layout[2].calculate_neuron_value(np.array([
            self.hidden_layout[0].value,
            self.hidden_layout[1].value,
            self.hidden_layout[2].value
        ]))

        return np.array([self.output_layout[0].value, self.output_layout[1].value, self.output_layout[2].value])

    def train(self, data, all_y_trues, learn_rate, loss_fun):
        epochs = 1000  # number of times to loop through the entire dataset

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):

                # --- Do a feedforward ---

                # For hidden layout
                self.hidden_layout[0].calculate_neuron_value(x)
                self.hidden_layout[1].calculate_neuron_value(x)
                self.hidden_layout[2].calculate_neuron_value(x)

                # For output layout
                self.output_layout[0].calculate_neuron_value(np.array([
                    self.hidden_layout[0].value,
                    self.hidden_layout[1].value,
                    self.hidden_layout[2].value,
                ]))

                self.output_layout[1].calculate_neuron_value(np.array([
                    self.hidden_layout[0].value,
                    self.hidden_layout[1].value,
                    self.hidden_layout[2].value,
                ]))

                self.output_layout[2].calculate_neuron_value(np.array([
                    self.hidden_layout[0].value,
                    self.hidden_layout[1].value,
                    self.hidden_layout[2].value,
                ]))

                y_pred = np.array([
                    self.output_layout[0].value,
                    self.output_layout[1].value,
                    self.output_layout[2].value
                ])

                # --- Calculate partial derivatives. ---
                # For example: d_L_d_w1 represents "partial L / partial w1"
                d_L_d_ypred = (-2 * (y_true - y_pred)).mean()

                # Neuron o1
                d_ypred_d_w10 = self.hidden_layout[0].value * deriv_sigmoid(self.output_layout[0].sum)
                d_ypred_d_w11 = self.hidden_layout[1].value * deriv_sigmoid(self.output_layout[0].sum)
                d_ypred_d_w12 = self.hidden_layout[2].value * deriv_sigmoid(self.output_layout[0].sum)
                d_ypred_d_b4 = deriv_sigmoid(self.output_layout[0].sum)

                # Neuron o2
                d_ypred_d_w13 = self.hidden_layout[0].value * deriv_sigmoid(self.output_layout[1].sum)
                d_ypred_d_w14 = self.hidden_layout[1].value * deriv_sigmoid(self.output_layout[1].sum)
                d_ypred_d_w15 = self.hidden_layout[2].value * deriv_sigmoid(self.output_layout[1].sum)
                d_ypred_d_b5 = deriv_sigmoid(self.output_layout[1].sum)

                # Neuron o3
                d_ypred_d_w16 = self.hidden_layout[0].value * deriv_sigmoid(self.output_layout[2].sum)
                d_ypred_d_w17 = self.hidden_layout[1].value * deriv_sigmoid(self.output_layout[2].sum)
                d_ypred_d_w18 = self.hidden_layout[2].value * deriv_sigmoid(self.output_layout[2].sum)
                d_ypred_d_b6 = deriv_sigmoid(self.output_layout[2].sum)


                d_ypred_d_h1 = np.sum(np.array([
                    self.output_layout[0].weights[0],
                    self.output_layout[1].weights[0],
                    self.output_layout[2].weights[0]]
                ) * np.array([
                    deriv_sigmoid(self.output_layout[0].sum),
                    deriv_sigmoid(self.output_layout[1].sum),
                    deriv_sigmoid(self.output_layout[2].sum)
                ]))
                d_ypred_d_h2 = np.sum(np.array([
                    self.output_layout[0].weights[1],
                    self.output_layout[1].weights[1],
                    self.output_layout[2].weights[1]]
                ) * np.array([
                    deriv_sigmoid(self.output_layout[0].sum),
                    deriv_sigmoid(self.output_layout[1].sum),
                    deriv_sigmoid(self.output_layout[2].sum)
                ]))
                d_ypred_d_h3 = np.sum(np.array([
                    self.output_layout[0].weights[2],
                    self.output_layout[1].weights[2],
                    self.output_layout[2].weights[2]]
                ) * np.array([
                    deriv_sigmoid(self.output_layout[0].sum),
                    deriv_sigmoid(self.output_layout[1].sum),
                    deriv_sigmoid(self.output_layout[2].sum)
                ]))

                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(self.hidden_layout[0].sum)
                d_h1_d_w2 = x[1] * deriv_sigmoid(self.hidden_layout[0].sum)
                d_h1_d_w3 = x[2] * deriv_sigmoid(self.hidden_layout[0].sum)
                d_h1_d_b1 = deriv_sigmoid(self.hidden_layout[0].sum)

                # Neuron h2
                d_h2_d_w4 = x[0] * deriv_sigmoid(self.hidden_layout[1].sum)
                d_h2_d_w5 = x[1] * deriv_sigmoid(self.hidden_layout[1].sum)
                d_h2_d_w6 = x[2] * deriv_sigmoid(self.hidden_layout[1].sum)
                d_h2_d_b2 = deriv_sigmoid(self.hidden_layout[1].sum)

                # Neuron h3
                d_h3_d_w7 = x[0] * deriv_sigmoid(self.hidden_layout[2].sum)
                d_h3_d_w8 = x[1] * deriv_sigmoid(self.hidden_layout[2].sum)
                d_h3_d_w9 = x[2] * deriv_sigmoid(self.hidden_layout[2].sum)
                d_h3_d_b3 = deriv_sigmoid(self.hidden_layout[2].sum)

                # --- Update weights and biases ---

                # First neuron from hidden_layout (h1)
                self.hidden_layout[0].weights[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.hidden_layout[0].weights[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.hidden_layout[0].weights[2] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w3
                self.hidden_layout[0].bias -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Second neuron from hidden_layout (h2)
                self.hidden_layout[1].weights[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.hidden_layout[1].weights[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w5
                self.hidden_layout[1].weights[2] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w6
                self.hidden_layout[1].bias -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Third neuron from hidden_layout (h3)
                self.hidden_layout[2].weights[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w7
                self.hidden_layout[2].weights[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w8
                self.hidden_layout[2].weights[2] -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w9
                self.hidden_layout[2].bias -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_b3

                # First neuron from output_layout (o1)
                self.output_layout[0].weights[0] -= learn_rate * d_L_d_ypred * d_ypred_d_w10
                self.output_layout[0].weights[1] -= learn_rate * d_L_d_ypred * d_ypred_d_w11
                self.output_layout[0].weights[2] -= learn_rate * d_L_d_ypred * d_ypred_d_w12
                self.output_layout[0].bias -= learn_rate * d_L_d_ypred * d_ypred_d_b4

                # Second neuron from output_layout (o2)
                self.output_layout[1].weights[0] -= learn_rate * d_L_d_ypred * d_ypred_d_w13
                self.output_layout[1].weights[1] -= learn_rate * d_L_d_ypred * d_ypred_d_w14
                self.output_layout[1].weights[2] -= learn_rate * d_L_d_ypred * d_ypred_d_w15
                self.output_layout[1].bias -= learn_rate * d_L_d_ypred * d_ypred_d_b5

                # Third neuron from output_layout (o3)
                self.output_layout[2].weights[0] -= learn_rate * d_L_d_ypred * d_ypred_d_w16
                self.output_layout[2].weights[1] -= learn_rate * d_L_d_ypred * d_ypred_d_w17
                self.output_layout[2].weights[2] -= learn_rate * d_L_d_ypred * d_ypred_d_w18
                self.output_layout[2].bias -= learn_rate * d_L_d_ypred * d_ypred_d_b6

            # --- Calculate total loss at the end of each epoch ---
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = loss_fun(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))
