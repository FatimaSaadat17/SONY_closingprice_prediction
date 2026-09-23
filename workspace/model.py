# import dependencies
import torch.nn as nn
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


# Create and define Pytorch dataset
class SONYStocksDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx, :, :], self.y[idx]




class SONYStocksDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, X_train, y_train, X_test, y_test, X_val, y_val):
        super().__init__()
        self.batch_size = batch_size
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val

    def setup(self, stage=None):
        self.train_dataset = SONYStocksDataset(self.X_train, self.y_train)
        self.test_dataset = SONYStocksDataset(self.X_test, self.y_test)
        self.val_dataset = SONYStocksDataset(self.X_val, self.y_val)
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False)




# create the LSTM model

class RegressionLSTM(nn.Module):
    def __init__(self, n_features, hidden_units=32, n_layers=1, dropout=0.2):
        super().__init__()
  # this is the number of features
        self.hidden_units = hidden_units
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_units, # dim of hidden state and cell states
            batch_first=True, #  PyTorch LSTM layer’s default is to use the second dimension instead. 
            #So we set batch_first=True to make the dimensions line up
            num_layers=self.n_layers, # number of lstm layers stacked vertically
            dropout=dropout if n_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        # ensures that GPU is optimized better and some weights are not fragmented or misaligned in blocks of memory preventing unneccessary jumps
        self.lstm.flatten_parameters()
        output, (h_n, c_n) = self.lstm(x)
        output = self.dropout(output[:, -1, :])  
        return self.linear(output)

''' nn.LSTM layer returns two elements:  out, (h_n, c_n) = self.lstm(x)
1) out - hidden state for every time step in the input sequence. Its shape is (Batch Size, Sequence Length, Hidden Size), same size as input batch and contains all hidden states 
2) This is a tuple containing the final hidden state (h_n) and the final cell state (c_n), which are the outputs of the last time step and the last layer.
 Since the last layer's output is needed, this is typically retrieved by indexing:  h_n[−1,:,:]

loss function - MSE
optimizer - Adam
early stopping - patience 15 '''


# define lightning module

class RegressionLSTMTrainer(pl.LightningModule):
    def __init__(self, n_features, hidden_units=32, n_layers=1, dropout=0.2):
        super().__init__()
        self.model = RegressionLSTM(n_features=n_features, hidden_units=hidden_units
                                    , n_layers=n_layers, dropout=dropout)
        self.hidden_units = hidden_units
        self.n_layers = n_layers
        self.dropout = dropout
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.model(x)
        return x
        
    def training_step(self, batch, batch_idx):
       x, y = batch
       y_hat = self(x)
  
       loss = self.criterion(y_hat, y)
  
       self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True)
       return loss
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        return self(x)
        
    # define the optimizers
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-3)
        return optimizer


    