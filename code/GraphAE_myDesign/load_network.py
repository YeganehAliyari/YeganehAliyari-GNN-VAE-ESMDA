import torch
import numpy as np
import Variational_GAE as graphAE
import graphAE_param as Param
import graphAE_dataloader as Dataloader
from datetime import datetime
from plyfile import PlyData
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from renderer.renderer import Mesh, Renderer
from matplotlib import cm
from torch.distributions import MultivariateNormal
import ESMDA as esmda
import time
import matplotlib.pyplot as plt


import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

param_enc = Param.Parameters()
param_dec = Param.Parameters()

param_enc.read_config("../../train/graphAE_Breast/encoder.config")
param_dec.read_config("../../train/graphAE_Breast/decoder.config")


# # param.augmented_data=Trues
param = param_enc
param.batch = 1  # this should be one since we are feeding data one-by-one in ESMDA

param.read_weight_path = "../../train/graphAE_Breast/weight_00/model_epoch0396.weight"
# # param.read_weight_path = "/home/mehrn/projects/def-mtavakol/mehrn/MeshConvolution_v2/code/old_weights/model_epoch0070.weight"

test_npy_fn = "../../data/Breast/data/test.npy"

# print("**********Initiate Netowrk**********")
model = graphAE.VariationalAutoencoder(param_enc, param_dec)
param = param_enc

print(model)
if param.read_weight_path != "":
    print("load " + param.read_weight_path)
    checkpoint = torch.load(param.read_weight_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.cuda()
    # input size is batch*number_point*3
    example = torch.rand((1,10859, 3)).cuda()
    # out put is a tuple of size 3 the first one is the output mesh with 
    # size batch*number_point*3 and the other two are latent variables with [1*26] 
 
    output=model(example)
    print(output[0].shape)




