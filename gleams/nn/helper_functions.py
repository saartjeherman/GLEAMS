import torch
import torch.nn as nn
import numpy as np

from gleams import config


class DummyModel(nn.Module):
    def __init__(self, filename: str = 'gleams.hdf5'):
        super(DummyModel, self).__init__()
        # Define the layers of your model here
        self.filename = filename
        self.siamese_model = None

        self.layer1 = nn.Linear(in_features=128, out_features=64)
        self.layer2 = nn.Linear(in_features=64, out_features=32)
        self.layer3 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
    




import torch
import torch.nn as nn
import torch.nn.functional as F

num_ref_spectra = config.num_ref_spectra
embedding_size = 32
num_precursor_features = config.num_precursor_features
num_fragment_features = config.num_fragment_features

class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.embedding_size = 32

        self.num_precursor_features =  num_precursor_features
        self.num_fragment_features = num_fragment_features
        self.num_ref_spectra_features = num_ref_spectra


        filters = 30
        kernel_size = 3
        strides = 1
        pool_size = 1
        pool_strides = 2
  

        # Precursor features dense layers
        self.precursor_dense_32 = nn.Sequential(
            nn.Linear(self.num_precursor_features, 32),
            nn.SELU()
        )
        self.precursor_dense_5 = nn.Sequential(
            nn.Linear(32, 5),
            nn.SELU()
        )

        # Fragment features convolutional blocks
        self.fragment_block_1 = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size, stride=strides),
            nn.SELU(),
            nn.Conv1d(filters, filters, kernel_size, stride=strides),
            nn.SELU(),
            nn.MaxPool1d(pool_size, stride=pool_strides)
        )
        
        self.fragment_block_2 = nn.Sequential(
            nn.Conv1d(filters, filters * 2, kernel_size, stride=strides),
            nn.SELU(),
            nn.Conv1d(filters * 2, filters * 2, kernel_size, stride=strides),
            nn.SELU(),
            nn.MaxPool1d(pool_size, stride=pool_strides)
        )

        self.fragment_block_3 = nn.Sequential(
            nn.Conv1d(filters * 2, filters * 4, kernel_size, stride=strides),
            nn.SELU(),
            nn.Conv1d(filters * 4, filters * 4, kernel_size, stride=strides),
            nn.SELU(),
            nn.Conv1d(filters * 4, filters * 4, kernel_size, stride=strides),
            nn.SELU(),
            nn.MaxPool1d(pool_size, stride=pool_strides)
        )

        self.fragment_block_4 = nn.Sequential(
            nn.Conv1d(filters * 4, filters * 8, kernel_size, stride=strides),
            nn.SELU(),
            nn.Conv1d(filters * 8, filters * 8, kernel_size, stride=strides),
            nn.SELU(),
            nn.Conv1d(filters * 8, filters * 8, kernel_size, stride=strides),
            nn.SELU(),
            nn.MaxPool1d(pool_size, stride=pool_strides)
        )

        self.fragment_block_5 = nn.Sequential(
            nn.Conv1d(filters * 8, filters * 8, kernel_size, stride=strides),
            nn.SELU(),
            nn.Conv1d(filters * 8, filters * 8, kernel_size, stride=strides),
            nn.SELU(),
            nn.Conv1d(filters * 8, filters * 8, kernel_size, stride=strides),
            nn.SELU(),
            nn.MaxPool1d(pool_size, stride=pool_strides)
        )

        self.fragment_flatten = nn.Flatten()

        # Reference spectra dense layers
        self.ref_spectra_dense_750 = nn.Sequential(
            nn.Linear(self.num_ref_spectra_features, 750),
            nn.SELU()
        )
        self.ref_spectra_output = nn.Sequential(
            nn.Linear(750, 250),
            nn.SELU()
        )

        # Final output layer
        self.output_layer = nn.Sequential(
            nn.Linear(5 + filters * 8 * (self.num_fragment_features // (pool_strides ** 5)) + 250, self.embedding_size),
            nn.SELU()
        )

    def forward(self, precursor_input, fragment_input, ref_spectra_input):
        # Process precursor input
        precursor_output = self.precursor_dense_32(precursor_input)
        precursor_output = self.precursor_dense_5(precursor_output)

        # Process fragment input
        print("fragment_input: ", fragment_input.shape)
        fragment_input = fragment_input.unsqueeze(1)  # Add channel dimension
        print("fragment_input: ", fragment_input.shape)
        fragment_output = self.fragment_block_1(fragment_input)
        fragment_output = self.fragment_block_2(fragment_output)
        fragment_output = self.fragment_block_3(fragment_output)
        fragment_output = self.fragment_block_4(fragment_output)
        fragment_output = self.fragment_block_5(fragment_output)
        fragment_output = self.fragment_flatten(fragment_output)

        # Process reference spectra input
        ref_spectra_output = self.ref_spectra_dense_750(ref_spectra_input)
        ref_spectra_output = self.ref_spectra_output(ref_spectra_output)

        # Combine all outputs
        combined_output = torch.cat((precursor_output, fragment_output, ref_spectra_output), dim=1)
        output = self.output_layer(combined_output)

        return output
    
    def initialize_dummy_weights(self):
        # Initialize all Linear layers with random values (or zeroes, as an example)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.weight, 0.01)  # Set all weights to a small constant
                nn.init.constant_(layer.bias, 0)  # Set all biases to zero

            elif isinstance(layer, nn.Conv1d):
                nn.init.constant_(layer.weight, 0.01)  # Set Conv1d weights to a small constant
                nn.init.constant_(layer.bias, 0)  # Set biases to zero
    

import numpy as np

def transfer_weights(keras_model, pytorch_model):
    keras_weights = keras_model.get_weights()
    pytorch_model.precursor_dense32.weight.data = torch.tensor(keras_weights[0].T)
    pytorch_model.precursor_dense32.bias.data = torch.tensor(keras_weights[1])
    pytorch_model.precursor_dense5.weight.data = torch.tensor(keras_weights[2].T)
    pytorch_model.precursor_dense5.bias.data = torch.tensor(keras_weights[3])
    # Transfer other layers similarly...
    # Note: You need to ensure the order of weights matches the PyTorch model's layers





# Instantiate the PyTorch model
pytorch_model = PyTorchModel()
#initiate model with dummy weights
pytorch_model.initialize_dummy_weights()
torch.save(pytorch_model.state_dict(), 'GLEAMS/data/gleams.pth')

#import keras
#keras_model = keras.models.load_model('GLEAMS/data/gleams.hdf5', compile=False)
#transfer_weights(keras_model, pytorch_model)

    