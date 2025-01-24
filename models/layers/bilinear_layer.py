import math
import torch
import torch.nn as nn
import torch.nn.init as init


class Bilinear(nn.Module):
    def __init__(self, input1_dim, input2_dim, bias=True):
        super(Bilinear, self).__init__()
        self.bilinear_weights = nn.Parameter(torch.rand(input1_dim, input2_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(input2_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def forward(self, input_1, input2):
        x = torch.matmul(input_1, self.bilinear_weights)
        output = torch.mul(x, input2.unsqueeze(1)) # (bs, time_step, dim) * (bs, 1, dim)
        if self.bias is not None:
            output += self.bias
        return output

    def reset_parameters(self):
        fan_in = self.bilinear_weights.size(0)  # Number of input features
        gain = math.sqrt(2.0)  # Default gain for ReLU activation
        std = gain / math.sqrt(fan_in)  # Standard deviation for Kaiming uniform

        # Manually create Kaiming uniform distribution for weights
        a = math.sqrt(3.0) * std  # Uniform bounds [-a, a]
        with torch.no_grad():
            self.bilinear_weights.data.uniform_(-a, a)  # Apply uniform initialization

        # Initialize bias using the same fan_in logic
        if self.bias is not None:
            bound = 1 / math.sqrt(fan_in)  # Compute bounds for bias
            with torch.no_grad():
                self.bias.data.uniform_(-bound, bound)  # Uniform initialization for bias