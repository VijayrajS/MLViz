from PIL import Image
import json

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

import torch
from torchvision.models import resnet18
from torchvision import transforms as T
from torchsummary import summary

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# a dict to store the activations
activation_list = defaultdict(list)
def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    activation_list[name].append(output.detach())
  return hook

# dynamic layer retrieval
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
    layer_obj = retrieve_layer(model, layer)
    if not activation:
        activation = layer

    hook = layer_obj.register_forward_hook(getActivation(activation))
    return hook

# register forward hooks on the layers of choice
print('***')
# print(vars(model));exit()

program_mode = 'multiple'
 # original model
model = resnet18(pretrained=True)
model = model.to(device)
model.eval()

if __name__ == "__main__" and program_mode == "single":

    # input (single)
    image = Image.open('./cat1.jpg')
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    X = transform(image).unsqueeze(dim=0).to(device)

    # input json
    layer_list = []
    with open('input.json') as f:
        layer_list = json.load(f)["layer_list"]

    hooks = []

    for layer in layer_list:
        if "activation" in layer.keys():
            hooks.append(register_hook(model, layer["layer_name"], layer["activation"]))
            continue
        hooks.append(register_hook(model, layer["layer_name"]))
    
    # forward pass -- getting the outputs
    out = model(X)

    # detach the hooks
    for i in range(len(hooks)):
        hooks[i].remove()

    print(activation.keys())
    print(activation['conv1'].shape)

    summary(model, (3, 224, 224))


    # a = np.random.random((16, 16))
    # plt.imshow(a, cmap='hot', interpolation='nearest')
    # plt.show()

def show_heatmap(data, title):
    sns.heatmap(data, cmap='plasma', cbar=True, xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.show()

if __name__ == "__main__" and program_mode == "multiple":
    inp_list = ["cat1.jpg", "cat2.jpg", "cat3.jpg", "dog1.jpg", "dog2.jpg", "dog3.jpg"]
    labels = []
    # input json
    layer_list = []
    with open('input.json') as f:
        layer_list = json.load(f)["layer_list"]
    
    hooks = []

    for layer in layer_list:
        if "activation" in layer.keys():
            hooks.append(register_hook(model, layer["layer_name"], layer["activation"]))
            continue
        hooks.append(register_hook(model, layer["layer_name"]))
    
    #* for X, y in dataloader := Dataloader(...

    for inp in inp_list: 
        image = Image.open(inp)
        transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        X = transform(image).unsqueeze(dim=0).to(device)
        out = model(X)
        labels.append(torch.argmax(out).item())
    
    # detach the hooks
    for i in range(len(hooks)):
        hooks[i].remove()
    
    print(labels)

    for k, v in activation_list.items():
        print(k, len(v), v[0].shape)

    # Average activation plot
    layer_of_interest = 'avgpool'
    # layer_of_interest = 'comp'
    for i in range(len(activation_list[layer_of_interest])):
        continue
        arr = np.squeeze(activation_list[layer_of_interest][i], axis=0)
        arr = np.squeeze(arr, axis=2).T
        # arr = arr[:, :, 0]
        # print(arr.shape);exit()
        
        sns.heatmap(arr, cmap='plasma', cbar=True, xticklabels=False, yticklabels=False)
        plt.title(f'Heatmap of {layer_of_interest} for {inp_list[i]}')
        plt.show()

    labels = [285, 285, 285, 153, 153, 153]
    
    # Class Averaged
    avg_285 = np.zeros(activation_list[layer_of_interest][0].shape)
    avg_153 = np.zeros(activation_list[layer_of_interest][0].shape)
    # avg_285 = np.zeros((14,14))
    # avg_153 = np.zeros((14,14))
    #! for label in set(labels): THIS SHOULD BE THE ACTUAL LOOP - TODO

    for i in range(len(activation_list[layer_of_interest])):
        arr = np.squeeze(activation_list[layer_of_interest][i], axis=0)
        arr = np.squeeze(arr, axis=2).T
        
        # arr = arr[:, :, 0]
        if labels[i] == 285:
            avg_285 = avg_285 + np.array(activation_list[layer_of_interest][i])
        else:
            avg_153 = avg_153 + np.array(activation_list[layer_of_interest][i])
    
    avg_285/=3
    avg_153/=3

    avg_285 = np.squeeze(avg_285, axis=0)
    avg_285 = np.squeeze(avg_285, axis=2).T
    avg_153 = np.squeeze(avg_153, axis=0)
    avg_153 = np.squeeze(avg_153, axis=2).T
    show_heatmap(avg_285, 'Average activations of class Cat')
    show_heatmap(avg_153, 'Average activations of class Dog')

    var = [np.squeeze(u.T[:,:,0]) for u in activation_list[layer_of_interest]]
    print(var[0].shape)
    var = np.array(var)
    var = np.var(var, axis=0)
    print(var.shape)
    show_heatmap(var, 'Variance across layer conv (slice 1)')
