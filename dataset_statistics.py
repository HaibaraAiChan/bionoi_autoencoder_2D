"""
Calculate and output mean and std of the image dataset
"""
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data
import copy
import argparse
from os import listdir
from os.path import isfile, join
from utils import UnsuperviseDataset
def getArgs():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-data_dir',
                        default='../bae-data-images/',
                        required=False,
                        help='directory of training images')
    parser.add_argument('-batch_size',
                        type=int,
                        default=128,
                        required=False,
                        help='the batch size, normally 2^n.')
    parser.add_argument('-num_data',
                        type=int,
                        default=50000,
                        required=False,
                        help='the batch size, normally 2^n.')                        

    return parser.parse_args()

def dataSetStatistics(data_dir, batch_size, num_data):
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Current device: '+str(device))

    transform = transforms.Compose([transforms.ToTensor()])
    img_list = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    dataset = UnsuperviseDataset(data_dir, img_list, transform=transform)  
    total = dataset.__len__()
    print('length of entire dataset:', total)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    # calculate mean and std for training data
    mean = 0.
    std = 0.
    m = 0
    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1) # reshape
        mean = mean + data.mean(2).sum(0)
        std = std + data.std(2).sum(0)
        m = m + batch_samples
        if m > num_data:
            break
    mean = mean / m
    std = std / m
    print('mean:',mean)
    print('std:',std)
    return mean, std

if __name__ == "__main__":
    args = getArgs()
    data_dir = args.data_dir
    batch_size = args.batch_size
    num_data = args.num_data
    dataSetStatistics(data_dir, batch_size,num_data)