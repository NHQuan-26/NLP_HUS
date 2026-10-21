import numpy as np


class SimpleRNNCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight matrices
        self.W_xh = np.random.randn(hidden_size, input_size)
        self.W_hh = np.random.randn(hidden_size, hidden_size)

        # Bias
        self.b_h = np.zeros(hidden_size)

    def forward(self, x_t, h_prev):
        h_t = np.tanh(self.W_xh @ x_t + self.W_hh @ h_prev + self.b_h)

        return h_t
