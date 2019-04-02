"""
Given the index of an image, take this image and feed it to a trained autoencoder,
then plot the original image and reconstructed image.
This code is used to visually verify the correctness of the autoencoder
"""
import torch
import argparse
import matplotlib.pyplot as plt
from utils import DenseAutoencoder, inference
from helper import imshow

def getArgs():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-data_dir',
                        default='../bae-data-images/',
                        required=False,
                        help='directory of training images')
    parser.add_argument('-feature_dir',
                        default='../bae-data-features/',
                        required=False,
                        help='directory of encoded')                        
    parser.add_argument('-model_file',
                        default='../trained_model/bionoi_autoencoder_dense.pt',
                        required=False,
                        help='file to save the model')
    return parser.parse_args()


if __name__ == "__main__":
    args = getArgs()
    index = args.index
    data_dir = args.data_dir
    model_file = args.model_file

 	# Detect if we have a GPU available
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print('Current device: '+str(device))   

    transform = transforms.Compose([transforms.ToTensor()])

	# put images into dataset
	img_list = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
	dataset = UnsuperviseDataset(data_dir, img_list, transform=transform)  

    	# create dataloader
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # instantiate model
	model = DenseAutoencoder(input_size, feature_size)
    # if there are multiple GPUs, split the batch to different GPUs
	if torch.cuda.device_count() > 1:
		print("Using "+str(torch.cuda.device_count())+" GPUs...")
		model = nn.DataParallel(model)
    # load model
	model.load_state_dict(torch.load(model_file))

    # get the reconstructed image
    reconstruct_image = inference(device, image, model)

    fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2,figsize=(14, 7))
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax2.imshow(reconstruct_image)
    ax2.set_title('Reconstructed Image')
    # show both figures
    #plt.savefig('./images/'+str(opMode)+'_'+str(imgClass)+str(index)+'.png')
    plt.show()