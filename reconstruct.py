"""
Given the index of an image, take this image and feed it to a trained autoencoder,
then plot the original image and reconstructed image.
This code is used to visually verify the correctness of the autoencoder
"""
import torch
import argparse
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import nn
from os import listdir
from os.path import isfile, join
from utils import UnsuperviseDataset, DenseAutoencoder, inference
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
    parser.add_argument('-model_file',
                        default='../trained_model/bionoi_autoencoder_dense.pt',
                        required=False,
                        help='file to save the model')
    parser.add_argument('-feature_size',
                        default=256,
						type=int,
                        required=False,
                        help='size of output feature of the autoencoder')                        
    return parser.parse_args()


if __name__ == "__main__":
    args = getArgs()
    index = args.index
    data_dir = args.data_dir
    model_file = args.model_file
    feature_size = args.feature_size

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Current device: '+str(device))   
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

    # instantiate model
    model = DenseAutoencoder(input_size, feature_size)
    # if there are multiple GPUs, split the batch to different GPUs
    #if torch.cuda.device_count() > 1:
    #    print("Using "+str(torch.cuda.device_count())+" GPUs...")
    #    model = nn.DataParallel(model)
    # load model
    model.load_state_dict(torch.load(model_file))

    # get the reconstructed image
    reconstruct_image = inference(device, image.unsqueeze(0), model)
    print('shape of reconstructed image:', reconstruct_image.shape)
    #print(reconstruct_image)

    # measure the loss between the 2 images
    criterion = nn.MSELoss()
    loss = criterion(image.unsqueeze(0).cpu(), reconstruct_image.cpu())
    print('loss between before and after:', loss)

    # plot images before and after reconstruction
    fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2,figsize=(14, 7))
    ax1.imshow(np.transpose(image.numpy(), (1,2,0)))
    ax1.set_title('Original Image')
    ax2.imshow(np.transpose(reconstruct_image.squeeze().detach().cpu().numpy(),(1,2,0)))
    ax2.set_title('Reconstructed Image')
    # show both figures
    #plt.savefig('./images/'+str(opMode)+'_'+str(imgClass)+str(index)+'.png')
    plt.show()