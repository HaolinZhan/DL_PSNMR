from DL_PSNMR import *
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import dataset
import h5py
import scipy.io as scio

#Use input data and network model from the corresponding folder,
# The name of the folder may differ from the serial number in the article
#Reading Underacquisition Data--.mat  Mat file saved in - v7.3 format in Matlab

path = r'text/figure3/Estradiol.mat'
matdata = h5py.File(path)
fn = 8192
FFT = np.zeros((1,fn))
FFTN = matdata['norm_origin_sp_real'][:].tolist()
fidfft = list(FFT)
fidfft_noise = list(FFTN)
test_list = np.c_[FFTN,FFT]
test_dataset =test_list

class MyDataset(dataset.Dataset):
    def __init__(self, data=None):
        self.data = data
        self.data_lengths = len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        src_data = data[:fn]
        trg_data = data[fn:]

        src_data = src_data.reshape(1,fn)
        trg_data = trg_data.reshape(1,fn)

        return src_data, trg_data

    def __len__(self):
        return self.data_lengths

def net_test():
    test_data = MyDataset(test_list)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=1,
                                              shuffle=False)
    net.eval()
    for step, (src_data, trg_data) in enumerate(test_loader):
        src_data = src_data.type(torch.FloatTensor)

        src_data = src_data

        output = net(src_data)
    output = output.cpu()
    output = output.detach().reshape(fn)

    plt.figure(1)
    plt.plot(output)
    plt.show()
#
if __name__ == "__main__":


    net =DL_PSNMR(1).cpu()
    # Load training model
    net.load_state_dict(torch.load('text/figure3/small_500.pt'))

    net_test()

