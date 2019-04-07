"""
Given the index of an image, take this image and feed it to a trained autoencoder,
then plot the original image and reconstructed image.
This code is used to visually verify the correctness of the autoencoder
"""
import torch
import argparse
import pickle
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import nn
from os import listdir
from os.path import isfile, join
from utils import UnsuperviseDataset, DenseAutoencoder, ConvAutoencoder, encode
from helper import imshow

def getArgs():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-index',
                        default=0,
                        type=int,
                        required=False,
                        help='index of image')
    parser.add_argument('-data_dir',
                        default='../bae-data-images/',
                        required=False,
                        help='directory of training images')
    parser.add_argument('-feature_dir',
                        default='../bae-data-features/',
                        required=False,
                        help='directory of generated features')
    parser.add_argument('-style',
                        default='conv',
                        required=False,
                        choices=['conv', 'dense'],
                        help='style of autoencoder')                        
    parser.add_argument('-normalize',
                        default=True,
                        required=False,
                        help='whether to normalize dataset')                       
    return parser.parse_args()

def feature_vec_gen(device, model, dataset, feature_dir):
    """
    Generate feature vectors for a single image
    """

    m = len(dataset)
    for i in range(m)
        image, file_name = dataset[i]
        file_name = file_name.split('.')
        file_name = file_name[0]
        feature_vec = encode(device, image, model)
        pickle_file = open(feature_dir + file_name + '.pickle','wb')
        pickle.dump(feature_vec, pklFile)

if __name__ == "__main__":
    args = getArgs()
    index = args.index
    data_dir = args.data_dir
    feature_dir = args.feature_dir
    style = args.style
    normalize = args.normalize

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Current device: '+str(device))   

    if style == 'conv':
        model_file = '../trained_model/bionoi_autoencoder_conv.pt'
    elif style == 'dense':
        model_file = '../trained_model/bionoi_autoencoder_dense.pt'

    data_mean = [0.6150, 0.4381, 0.6450]
    data_std = [0.6150, 0.4381, 0.6450]   
    if normalize == True:
        print('normalizing data:')
        print('mean:', data_mean)
        print('std:', data_std)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((data_mean[0], data_mean[1], data_mean[2]),
                                                             (data_std[0], data_std[1], data_std[2]))])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
	
    # put images into dataset
    img_list = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    dataset = UnsuperviseDataset(data_dir, img_list, transform=transform)  
    image, file_name = dataset[index]
    # calulate the input size (flattened)
    print('name of input:', file_name)
    image_shape = image.shape
    print('shape of input:', image_shape)
    input_size = image_shape[0]*image_shape[1]*image_shape[2]
    print('flattened input size:',input_size) 

    # instantiate and load model
    if style == 'conv':
        model = ConvAutoencoder()
    elif style == 'dense':
        model = DenseAutoencoder(input_size, feature_size)  
 
    # if there are multiple GPUs, split the batch to different GPUs
    if torch.cuda.device_count() > 1:
        print("Using "+str(torch.cuda.device_count())+" GPUs...")
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_file))

    # generate features for images in data_dir
    feature_vec_gen(device, model, dataset, feature_dir)