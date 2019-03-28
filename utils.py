"""
Functions and classes to build the autoencoder for bionoi
"""

import torch
import time
import copy
import numpy as np
from skimage import io
from torch import nn
from torch.utils import data
import matplotlib.pyplot as plt 

class UnsuperviseDataset(data.Dataset):
	"""
	A data set containing images for unsupervised learning. All the images are stored in one single folder,
	and there are no labels
	"""
	def __init__(self, data_dir, filename_list, transform=None):
		self.data_dir = data_dir
		self.filename_list = filename_list
		self.transform = transform

	def __len__(self):
		return len(self.filename_list)

	def __getitem__(self, index):
		# indexing the file
		f = self.filename_list[index]
		# load file
		#print("index:",index)
		#print(self.data_dir+f)
		img = io.imread(self.data_dir+f)
		# apply the transform
		return self.transform(img)

class DenseAutoencoder(nn.Module):
	"""
	A vanilla-style autoencoder, takes 2d images as input, then the
	flattened vector is sent to the autoencoder, which is composed of
	multiple dense layers.
	"""
	def __init__(self, input_size, feature_size):
		"""
		input_size -- int, flattened input size
		feature_size -- int
		"""
		super(DenseAutoencoder, self).__init__()
		self.enc_fc1 = nn.Linear(input_size,256)
		self.enc_fc2 = nn.Linear(256,feature_size)		
		self.dec_fc1 = nn.Linear(feature_size,256)
		self.dec_fc2 = nn.Linear(256,input_size)

		self.relu = nn.LeakyReLU(0.1)
		self.sigmoid = nn.Sigmoid()
		self.dropout = nn.Dropout(0.5)		

	def flatten(self, x):
		x = x.view(x.size(0), -1)
		return x

	def encode(self, x):
		x = self.flatten(x)
		x = self.relu(self.enc_fc1(x))
		x = self.relu(self.enc_fc2(x))
		return x

	def decode(self, x):
		x = self.relu(self.dec_fc1(x))
		x = self.sigmoid(self.dec_fc2(x))
		x = self.unflatten(x)
		return x

	def unflatten(self, x):
		x = torch.reshape(x,(x.size(0),3,256,256))
		return x

	def forward(self, x):
		x = self.encode(x)
		x = self.decode(x)
		return x

def train(device, num_epochs, dataloader, model, criterion, optimizer, learningRateScheduler):
	since = time.time()
	train_loss_history = []

	# send model to GPU if available
	model.to(device)

	# need a deep copy here because weights will be updated in the future
	best_model_wts = copy.deepcopy(model.state_dict())
	best_loss = float("inf") # initial best loss
	for epoch in range(num_epochs):
		print(' ')
		print('Epoch {}/{}'.format(epoch+1, num_epochs))
		print('-' * 15)		
		running_loss = 0.0
		for images in dataloader:
			images.to(device) # send to GPU if available
			images_out = model(images)# forward
			images = images.cpu()
			images_out = images_out.cpu()
			loss = criterion(images,images_out)
			loss.backward() # back propagation
			optimizer.step() # updata parameters
			optimizer.zero_grad() # zero out the gradients
			running_loss += loss.item() * images.size(0) # accumulate loss for this epoch
		epoch_loss = running_loss / len(dataloader.dataset)
		print('Training loss:{:4f}'.format(epoch_loss))
		train_loss_history.append(epoch_loss) # store loss of current epoch
		if epoch_loss <= best_loss:
			best_loss = epoch_loss
			best_model_wts = copy.deepcopy(model.state_dict())
		
	time_elapsed = time.time() - since
	print( 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print( 'Best training loss: {:4f}'.format(best_loss))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model, train_loss_history