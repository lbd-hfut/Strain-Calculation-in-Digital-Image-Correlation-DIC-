import torch
import torch.nn as nn
import numpy as np

class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.width = layers[1:-1]
        self.num_layers = len(self.width)
        self.input_size = layers[0]
        self.output_size = layers[-1]
        
        # Define input layer
        self.input_layer = nn.Linear(self.input_size, self.width[0])
        
        # Define hidden layers (MLP)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.width[i], self.width[i+1]) for i in range(self.num_layers-1)]
            )
        
        # Define output layer
        self.output_layer = nn.Linear(self.width[-1], self.output_size)
        
        
        # Define activation function parameter 'a'
        self.a = nn.Parameter(torch.tensor([0.2] * (self.num_layers + 2)))

    def forward(self, x):
        
        # Input layer
        x = self.input_layer(x)
        x = 5 * self.a[0] * x
        x = torch.tanh(x)
        
        # Hidden layers (MLP)
        for i in range(self.num_layers-1):
            x = self.hidden_layers[i](x)
            x = 5 * self.a[i + 1] * x
            x = torch.tanh(x)
        
        # Output layer
        x = 5 * self.a[-1] * x
        x = self.output_layer(x)
        
        return x

class PositionalEncoding:
    def __init__(self, num_frequencies, input_dims=3):
        self.num_frequencies = num_frequencies
        self.input_dims = input_dims
        self.create_encoding_functions()
        
    def create_encoding_functions(self):
        # Define the frequency bands
        self.frequency_bands = 2 ** torch.linspace(0, self.num_frequencies - 1, self.num_frequencies)
        
        # Create the list of encoding functions
        self.encoding_functions = []
        for freq in self.frequency_bands:
            self.encoding_functions.append(lambda x, freq=freq: torch.sin(2 * np.pi * freq * x))
            self.encoding_functions.append(lambda x, freq=freq: torch.cos(2 * np.pi * freq * x))

    def encode(self, x):
        # x is expected to be of shape (N, input_dims) where N is the batch size
        encodings = [x]  # Start with the original input
        for fn in self.encoding_functions:
            encodings.append(fn(x))
        return torch.cat(encodings, dim=-1)

# # Example usage
# # Input is a 3D coordinate (e.g., x, y, z)
# positions = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # Example 3D positions

# # Positional encoding with L = 10 (as mentioned in the paper)
# encoder = PositionalEncoding(num_frequencies=10, input_dims=3)
# encoded_positions = encoder.encode(positions)

# print(encoded_positions.shape)  # Should be [N, 3 * (2 * 10)]
# print(encoded_positions)        # Encoded values for the positions
