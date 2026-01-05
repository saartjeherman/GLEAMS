import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
import logging
from typing import List

import config
from nn import data_generator


logger = logging.getLogger('gleams')


def euclidean_distance(xy: torch.Tensor) -> torch.Tensor:
    """
    Euclidean distance between two vectors using torch.

    Parameters
    ----------
    vectors : torch.Tensor
        The input vectors.

    Returns
    -------
    torch.Tensor
        The Euclidean distances between the input vectors.
    """
    x, y = xy
    sum_square = torch.sum(torch.square(x - y), dim=1)
    return torch.sqrt(torch.maximum(sum_square, torch.tensor(1e-12)))
    

def eucl_dist_output_shape(shapes):
    """
    Get the shape of the Euclidean distance output.

    Parameters
    ----------
    shapes
        Input shapes to the Euclidean distance calculation.

    Returns
    -------
    The shape of the Euclidean distance output.
    """
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Contrastive loss function adapted from Hadsell et al. 2006.
    (http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)

    The contrastive loss is modified so that it takes a certainty that labels
    are correct into account. This helps the neural network to overcome
    incorrectly labeled instances.

    Parameters
    ----------
    y_true : torch.Tensor
        The true class labels.
    y_pred : torch.Tensor
        The predicted embedded Euclidean distances.

    Returns
    -------
    The contrastive loss between the true and predicted class labels.
    """
    ramp_square = torch.square(torch.minimum(y_pred, torch.tensor(config.margin)))
    margin_square = torch.square(torch.maximum(torch.tensor(config.margin) - y_pred, torch.tensor(0.0)))
    y_true = y_true.float()
    return torch.mean(y_true * config.loss_label_certainty * ramp_square +
                      (1 - y_true * config.loss_label_certainty) * margin_square)



class Embedder(nn.Module):
    """
    A spectrum embedder formed by a Siamese neural network.
    """

    def __init__(self, num_precursor_features: int, num_fragment_features: int,
                 num_ref_spectra_features: int, lr: float,
                 filename: str = 'gleams.pth'):
        """
        Instantiate the Embedder based on the given number of input features.

        Parameters
        ----------
        num_precursor_features : int
            The number of input precursor features.
        num_fragment_features : int
            The number of input fragment features.
        num_ref_spectra_features : int
            The number of input reference spectra features.
        lr : float
            The learning rate for the Adam optimizer.
        filename : str
            Filename to save the trained PyTorch model.
        """
        super(Embedder, self).__init__()
        self.num_precursor_features = num_precursor_features
        self.num_fragment_features = num_fragment_features
        self.num_ref_spectra_features = num_ref_spectra_features
        self.embedding_size = config.embedding_size

        self.lr = lr
        self.filename = filename

        self.siamese_model = None
        self.num_gpu = torch.cuda.device_count()

        filters = 30
        kernel_size = 3
        strides = 1
        pool_size = 1
        pool_strides = 2

        self.batch_size = config.batch_size
  

        # Precursor features dense layers
        self.precursor_dense_32 = nn.Sequential(
            nn.Linear(num_precursor_features, 32),
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

        #self.fragment_flatten = nn.Flatten()

        # Reference spectra dense layers
        self.ref_spectra_dense_750 = nn.Sequential(
            nn.Linear(num_ref_spectra_features, 750),
            nn.SELU()
        )
        self.ref_spectra_output = nn.Sequential(
            nn.Linear(750, 250),
            nn.SELU()
        )

        # Final output layer
        self.output_layer = nn.Sequential(
            nn.Linear(5 + filters * 8 * 71 + 250, self.embedding_size),
            nn.SELU()
        )



    def forward(self, precursor_input, fragment_input, ref_spectra_input):
        # Process precursor input
        precursor_output = self.precursor_dense_32(precursor_input)
        precursor_output = self.precursor_dense_5(precursor_output)

        # Process fragment input
        fragment_input = fragment_input.unsqueeze(0)
        fragment_output = self.fragment_block_1(fragment_input)
        fragment_output = self.fragment_block_2(fragment_output)
        fragment_output = self.fragment_block_3(fragment_output)
        fragment_output = self.fragment_block_4(fragment_output)
        fragment_output = self.fragment_block_5(fragment_output) #shape [240,71]
        fragment_output = fragment_output.flatten()    #shape [17040]

        # Process reference spectra input
        ref_spectra_output = self.ref_spectra_dense_750(ref_spectra_input)
        ref_spectra_output = self.ref_spectra_output(ref_spectra_output)


        # Expand dimensions to match the batch size
        precursor_output = precursor_output.unsqueeze(0)
        fragment_output = fragment_output.unsqueeze(0)
        ref_spectra_output = ref_spectra_output.unsqueeze(0)

        # Combine all outputs
        combined_output = torch.cat((precursor_output, fragment_output, ref_spectra_output), dim=1)
        output = self.output_layer(combined_output)

        return output
    
    def _get_embedder_model(self):
        """
        Get the base embedder model (i.e. a single arm of the Siamese model).

        Returns
        -------
        Model
            The embedder model.
        """
        if self.siamese_model is None:
            raise ValueError('The embedder model has not been constructed yet')
        else:
            return self
        

    def save(self) -> None:
        """
        Save the embedder model's weights.
        """
        if self.siamese_model is None:
            raise ValueError('The embedder model has not been constructed yet')
        else:
            torch.save(self.state_dict(), self.filename)

    def load(self) -> None:
        """
        Load a previously trained Embedder model.
        """
        state_dict = torch.load(self.filename)
        model_dict = self.state_dict()

        # Filter out mismatched keys
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(filtered_dict)

        # Load the updated state_dict
        self.load_state_dict(model_dict)
        #self.load_state_dict(torch.load(self.filename))

        self.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def _build_embedder_model(self):
        """
        Build the embedder model.

        Returns
        -------
        Model
            The embedder model.
        """
        return self
    
    def build(self) -> None:
        """
        Build the Siamese model.
        """
        self.siamese_model = self._build_embedder_model()
        self.siamese_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def train_model(self, train_loader: DataLoader, num_epochs: int = 1, validators: list = None) -> None:
        """
        Train the Siamese model.

        Parameters
        ----------
        train_loader : DataLoader
            The training data loader.
        num_epochs : int
            The number of epochs for which training occurs.
        validators : list
            The validation data loaders.
        """
        self.train()
        for epoch in range(num_epochs):
            for batch in train_loader:
                precursor_input, fragment_input, ref_spectra_input, labels = batch
                self.optimizer.zero_grad()
                outputs = self.forward(precursor_input, fragment_input, ref_spectra_input)
                loss = contrastive_loss(outputs, labels)
                loss.backward()
                self.optimizer.step()
            if validators:
                self.validate(validators)

    def validate(self, validators: list) -> None:
        """
        Validate the model on the validation datasets.

        Parameters
        ----------
        validators : list
            The validation data loaders.
        """
        self.eval()
        with torch.no_grad():
            for val_loader in validators:
                for batch in val_loader:
                    precursor_input, fragment_input, ref_spectra_input, labels = batch
                    outputs = self.forward(precursor_input, fragment_input, ref_spectra_input)
                    loss = contrastive_loss(outputs, labels)
                    print(f'Validation loss: {loss.item()}')

    def embed(self, encodings_generator: data_generator.EncodingsSequence) -> np.ndarray:
        """
        Transform samples using the embedder model.

        Parameters
        ----------
        encodings_generator: data_generator.EncodingsSequence
            A generator that gives the input samples as batches of a list of
            length three representing the precursor features, fragment
            features, and reference spectra features.

        Returns
        -------
        np.ndarray
            The embeddings of the given samples.
        """
        # Set the model to evaluation mode
        self.eval()

        embeddings = []
        
        for batch in encodings_generator:  
            # print the dimensions of the batch
            precursor_inputs = batch[0]
            fragment_inputs = batch[1]
            ref_spectra_inputs = batch[2]
            batch_size = precursor_inputs.shape[0]
            for i in range(batch_size):  
                precursor_input = precursor_inputs[i]
                fragment_input = fragment_inputs[i]
                ref_spectra_input = ref_spectra_inputs[i]
                
                # Convert inputs to tensors (assuming numpy arrays are passed)
                precursor_input = torch.tensor(precursor_input, dtype=torch.float32)
                fragment_input = torch.tensor(fragment_input, dtype=torch.float32)
                ref_spectra_input = torch.tensor(ref_spectra_input, dtype=torch.float32)

                # Move tensors to the device
                precursor_input = precursor_input.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                fragment_input = fragment_input.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                ref_spectra_input = ref_spectra_input.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            
                # Pass the inputs through the model
                with torch.no_grad():  # Disable gradient tracking for inference
                    output = self.forward(precursor_input, fragment_input, ref_spectra_input)
                    embeddings.append(output.cpu().tolist())    
                

        # Concatenate all embeddings into a single numpy array
        return np.concatenate(embeddings, axis=0)

        #new_array = np.zeros((968, 32))
        #return new_array


