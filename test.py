import utils
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X_C_GNN, X_P_GNN, X_T_GNN, X_Ext_GNN, X_C_ST, X_P_ST, X_T_ST):
        """
        Args:
            datasets = [X_C_GNN, X_P_GNN, X_T_GNN, X_Ext_GNN]
        """
        self._X_C_GNN = X_C_GNN
        self._X_P_GNN = X_P_GNN
        self._X_T_GNN = X_T_GNN
        self._X_Ext_GNN = X_Ext_GNN
        self._X_C_ST = X_C_ST
        self._X_P_ST = X_P_ST
        self._X_T_ST = X_T_ST

    def __len__(self):
        return len(self._X_C_GNN)

    def __getitem__(self, idx):
        sample = [self._X_C_GNN[idx], self._X_P_GNN[idx], self._X_T_GNN[idx], self._X_Ext_GNN[idx],
                  self._X_C_ST[idx], self._X_P_ST[idx], self._X_T_ST[idx]]

        return sample


def test_1():
    stop_distribution = utils.stop_location_reader('stops_locations.txt')

    for key, item in stop_distribution.items():
        print(key, item)

    print(len(stop_distribution))

def test_2():
    stop_distribution_BSID = utils.stop_distribution_reader('BSID_top_list_10.txt')
    print(stop_distribution_BSID)
    print(len(stop_distribution_BSID))


def test_3():
    t1 = torch.zeros(100, 3, 2, 10, 10)
    print(len(t1))

def test_4():



if __name__ == '__main__':
    test_3()
