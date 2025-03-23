import torch
from torchsummary import summary

from collections import defaultdict

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ActivationExtractor:
	def __init__(self, model):
		self.model = model
		self.activations = defaultdict(list)

	def getActivation(name):
		# the hook signature
		def hook(input, output):
			activation[name] = output.detach()
		
		return hook
		
	def retrieve_layer(model, layer_str):
		layer_str_list = layer_str.split('.')
		current_obj = model

		for name in layer_str_list:
			name_arr = name.split('[')
			index = None

			if len(name_arr) > 1:
				index = int(name_arr[-1][:-1])
				name = name_arr[0]
			
			current_obj = getattr(current_obj, name)
			if index is not None:
				current_obj = current_obj[index]
		
		return current_obj

	# dynamic hook registration
	def register_hook(model, layer, activation=None):
		layer_obj = retrieve_layer(self.model, layer)
		if not activation:
			activation = layer

		hook = layer_obj.register_forward_hook(getActivation(activation))
		return hook
	
	def constructActivations(string):
		
