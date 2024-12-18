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

class SinActivation(nn.Module):
    def __init__(self):
        super(SinActivation, self).__init__()
    def forward(self, x):
        return torch.sin(1*x)
    
class PhiActivation(nn.Module):
    def __init__(self):
        super(PhiActivation, self).__init__()
        self.relu = nn.ReLU()
    def forward(self, x):
        y = self.relu(x)**2 - 3*self.relu(x-1)**2 + 3*self.relu(x-2)**2 - self.relu(x-3)**2
        return y
    
class SeReLU(nn.Module):
    def __init__(self):
        super(SeReLU, self).__init__()
        self.relu = nn.ReLU()
    def forward(self, x):
        y = self.relu(x) * self.relu(1-x)
        return y
    
class SelfAdaptive(nn.Module):
    def __init__(self):
        super(SelfAdaptive, self).__init__()
        self.a = nn.Parameter(torch.tensor(0.2))
    def forward(self, x):
        y = 5 * self.a * x
        return y

# Define activation functions
activation_dict = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'serelu': SeReLU(),
    'softplus': nn.Softplus(),
    'sin': SinActivation(),
    'phi': PhiActivation(),
}

class MscaleDNN(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim, scales, activation):
        super(MscaleDNN, self).__init__()
        self.scales = scales
        self.activation = activation_dict[activation]
        # self.activation = [activation_dict["phi"], activation_dict["phi"], activation_dict["phi"], activation_dict["phi"]]
        self.subnets = nn.ModuleList()
        
        for scale in scales:
            layers = []
            prev_units = input_dim
            for i, units in enumerate(hidden_units):
                layers.append(nn.Linear(prev_units, units))
                layers.append(SelfAdaptive())
                layers.append(self.activation)
                # layers.append(self.activation[i])
                prev_units = units
            layers.append(SelfAdaptive())
            layers.append(nn.Linear(prev_units, output_dim))
            self.subnets.append(nn.Sequential(*layers))

    def forward(self, x):
        outputs = []
        for i, scale in enumerate(self.scales):
            scaled_x = x * scale
            outputs.append(self.subnets[i](scaled_x))
        return torch.mean(torch.stack(outputs), dim=0)
    
    # Set the parameters for the early stop method
    def Earlystop_set(self, patience=10, delta=0, path=None):
        self.patience = int(patience)
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        
    # Logical judgment of executing the early stop method
    def Earlystop(self, val_loss, model, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss < val_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
    # Save the checkpoint of the current optimal model
    def save_checkpoint(self, val_loss, model):
        if self.path:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'loss': val_loss
            }
            torch.save(checkpoint, self.path)
            self.best_model = model.state_dict()
            
    # Freeze all trainable parameters  
    def freeze_all_parameters(self):  
        for param in self.parameters():  
            param.requires_grad = False  
  
    # Unfreeze all trainable parameters  
    def unfreeze_all_parameters(self):  
        for param in self.parameters():  
            param.requires_grad = True


# # Example usage
# # Input is a 3D coordinate (e.g., x, y, z)
# positions = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # Example 3D positions

# # Positional encoding with L = 10 (as mentioned in the paper)
# encoder = PositionalEncoding(num_frequencies=10, input_dims=3)
# encoded_positions = encoder.encode(positions)

# print(encoded_positions.shape)  # Should be [N, 3 * (2 * 10)]
# print(encoded_positions)        # Encoded values for the positions


# # Example usage
# input_dim = 2          
# hidden_units = [64, 32] 
# output_dim = 1          
# scales = [0.5, 1.0, 2.0]  
# activation = 'sin'     

# model = MscaleDNN(input_dim, hidden_units, output_dim, scales, activation)

# print(model)

# x = torch.randn(4, input_dim)

# output = model(x)

# print("Model output:")
# print(output)