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


def get_faces_from_ply(ply):
    faces_raw = ply['face']['vertex_index']
    faces = np.zeros((faces_raw.shape[0], 3)).astype(np.int32)
    for i in range(faces_raw.shape[0]):
        faces[i][0]=faces_raw[i][0]
        faces[i][1]=faces_raw[i][1]
        faces[i][2]=faces_raw[i][2]
    
    
    return faces
    




# Load weights 
def Load(param_enc,param_dec,test_npy_fn,test_npy_gt_fn, out_ply_folder, out_img_folder, is_render_mesh=False, skip_frames =0):
    
    # For now the test_npy_fn is similar to test_npy_gt

    
   
    print ("**********Initiate Netowrk**********")
    model = graphAE.VariationalAutoencoder(param_enc,param_dec)
    param = param_enc
    if(param.read_weight_path!=""):
        print ("load "+param.read_weight_path)
        checkpoint = torch.load(param.read_weight_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        #model.init_test_mode()
    
    model.cuda()
    model.eval()
    
    template_plydata = PlyData.read(param.template_ply_fn)
    faces = get_faces_from_ply(template_plydata)
    
    print ("**********Get test pcs**********", test_npy_fn)
    ##get ply file lst
    pc_lst= np.load(test_npy_fn)
    pc_gt_lst= np.load(test_npy_gt_fn)


    # print (pc_lst.shape[0], "meshes in total.")
    pc_lst[:,:,0:3] -= pc_lst[:,:,0:3].mean(1).reshape((-1,1,3)).repeat(param.point_num, 1)
    pc_gt_lst[:,:,0:3] -= pc_gt_lst[:,:,0:3].mean(1).reshape((-1,1,3)).repeat(param.point_num, 1)

    pc_lst = torch.FloatTensor(pc_lst)
    pc_gt_lst = torch.FloatTensor(pc_gt_lst)
    
    return   model, pc_lst, pc_gt_lst



def measurment_function (model, Latent_ensambels, surface_index):

    surface_index = surface_index.cuda()
    out_put_ensamble_mesh =  model.decoder(Latent_ensambels)
    out_put_surface_nodes = out_put_ensamble_mesh[:,surface_index,:]
    return out_put_surface_nodes
    
def encoder_function (model,N_ensambels, pc_lst_sample):
    # a = torch.FloatTensor(pc_lst_sample)
    pc_lst_sample = torch.unsqueeze(pc_lst_sample,0).cuda()
    latent_mu, latent_var =  model.encoder(pc_lst_sample)
    covarinace = torch.diag(torch.squeeze((0.5*latent_var).exp(),0))
    m = MultivariateNormal(torch.squeeze(latent_mu,0), covarinace)
    Latent_ensambels = m.sample(torch.Size([N_ensambels]))
    return Latent_ensambels



param_enc=Param.Parameters()
param_dec=Param.Parameters()
param_enc.read_config("../../train/graphAE_Breast/encoder.config")
param_dec.read_config("../../train/graphAE_Breast/decoder.config") 

#param.augmented_data=True
param = param_enc
param.batch =1  # this should be one since we are feeding data one-by-one in ESMDA 

param.read_weight_path = "../../train/graphAE_Breast/weight_00/model_epoch0372.weight"
print (param.read_weight_path)

test_npy_fn = "../../data/Breast/test.npy"
test_npy_gt_fn = "../../data/Breast/test.npy"
surface_indeces_fn = "../../data/Breast/index.npy"

out_test_folder = "../../train/graphAE_Breast/ESMDA/"

out_ply_folder = out_test_folder
out_img_folder = out_test_folder+"img/"

out_weight_visualize_folder=out_test_folder+"weight_vis/"

out_interpolation_folder = out_test_folder+"interpolation/"

# LOad weights and load data sets 

model, pc_lst, pc_gt_lst = Load(param_enc,param_dec,test_npy_fn,test_npy_gt_fn, out_ply_folder, out_img_folder)


# Build output functions 


# Apply ESMDA 

# Define the ESMDA parameters 
Na = 5 # Number of esmda iterations 
num=2
N_ensambels = 1
#surface_indeces= torch.tensor(np.load(surface_indeces_fn),dtype=torch.long)

# surface_indeces = torch.tensor([1784,1243,1484,1149,959,1595,386,1680,2198,779,239,809,1930,968,1402,1994,1087,149,118,770,1151,1178,
#                                 1323,742,1539,932,201,407,1125,1573,422,559,1274,1649,2045,229,1610,2186,918,2301,2218,1138,850,2172,
#                                 1462,2195,624,251,1531,1086,744,1464,1071,136,2050,1247,2281,1238,843,623,75,380,352,1939,221,1874,1517,
#                                 1466,782,1842,232,1675,856,727,626,847,1179,1108,1260,1633,2282,2322,2217,1070,984,1094,384,1372,1546,
#                                 2170,1209,2256,1129,2137,122,2243,1678,1373,1302,745,1806,1616,1016,2020,2033,94,2060,22,2004,1292,1916,1050,
#                                 1063,547,1267,1265,1985,1559,650,2145,1093,1611,581,73,673,2365,1777,1881,442,1085,357,2292,1366,919,
#                                 1401,628,443,105,1722,2202,1324,134,344,2135,381,1448,1659,1509,1312,1089,1077,1158,194,1014,928,1444,
#                                 817,524,1479,1171,1709,1133,683,1711,1582,1468,257,2119,1241,1641,697,2019,2296,1864,2359,2038,1840,2320,
#                                 1669,1795,696,1706,572,2283,1703,1416,2305,1263,2073,584,1105,276,1811,1362,1215,1029,1004,1442,2174,2368,
#                                 2289,1097,1427,1096,2088,1199,1868,2022,1080,1503,2245,1470,1715,1201,1172,1233,1410,1545,1275,1127,980,1426,
#                                 1264,1409,1904,2274,1574,1508],dtype=torch.long)

surface_indeces = torch.tensor([891,898,901,904,906,913,914,915,919,926,929,931,939,940,941,942,944,964,971,976,978,979,980,989,995,997,999,1001,
                                1011,1018,1024,1025,1026,1032,1034,1036,1038,1039,1040,1042,1065,1067,1069,1076,1079,1080,1083,1084,1088,1094,1095,
                                1099,1104,1105,1107,1136,1138,1146,1147,1157,1158,1161,1166,1167,1171,1176,1198,1205,1213,1221,1230,1231,1233,1234,
                                1242,1251,1252,1264,1271,1277,1278,1288,1290,1295,1297,1307,1310,1312,1316,1343,1345,1356,1357,1363,1371,1379,1384,
                                1388,1393,1394,1401,1409,1412,1422,1427,1455,1458,1463,1467,1468,1473,1477,1493,1506,1512,1513,1528,1541,1543,1559,
                                1577,2401,2402,2403,2407,2434,2435,2443,2457,2464,2473,2493,2501,2516,2528,2531,2538,2550,2569,2578,2579,2586,2594,
                                2609,2648,2654,2663,2666,2671,2673,2674,2698,2703,2706,2707,2717,2721,2726,2727,2731,2756,2777,2781,2806,2807,2818,
                                2832,2844,2882,2889,2922,2932,2939,2950,2959,2960,2975,2976,2987,3013,3025,3029,3047,3051,3112,3118,3147,3148,3193,
                                3197,3200,3249,3260,3261,3291,3306,3360,3371,3373,3384,3394,3418,3423,3424,3448,3468,3488,3501,3513,3543,3584,3594,
                                3602,3683,3701,3750,3756,3775,3810,3817,3829,3874,3880,3881,3888,3891,3894,3895,3926,3934,3968,3972,3987,3990,3992,
                                4015,4036,4073,4152,4175,4184,4185,4216,4225,4242,4267,4304,4336,4356,4368,4369,4402,4433,4437,4451,4470,4514,4534,
                                4535,4537,4587,4591,4643,4646,4665,4670,4693,4694,4729,4735,4773,4794,4795,4855,4865,4896,4900,4923,4954,4955,4967,
                                5006,5021,5062,5152,5196,5261,5268,5290,5291,5342,5351,5404,5492,5499,5500,5549,5597,5702,5746,5822,5841,5865,5871,
                                5877,5957,6028,6052,6251,6297,6301,6303,6326,6382,6392,6445,6553,6574,6605,6699,6700,6717,6753,6768,6779,6830,6945,
                                6966,6998,7096,7098,7144,7230,7244,7277,7333,7406,7423,7445,7540,7641,7680,7980,8016,8023,8042,8147,8253,8278,8300,
                                8310,8313,8323,8344,8375,8379,8385,8453,8479,8505,8537,8583,8603,8616,8621,8660,8662,8675,8717,8722,8745,8759,8769,
                                8801,8802,8813,8825,8843,8845,8857,8872,8932,9018,9039,9087,9139,9187,9188,9190,9244,9278,9308,9329,9342,9345,9379,
                                9420,9431,9435,9444,9461,9483,9551,9587,9589,9591,9593,9623,9628,9650,9688,9708,9728,9747,9755,9764,9777,9785,9796,
                                9806,9824,9840,9879,9884,9898,9907,9913,9916,9988,9990,9998,10023,10122,10171,10281,10303,10307,10309,10319,10377,
                                10409,10479,10530,10532,10573,10590,10673,10686,10719,10755,10778,10791,10808],dtype=torch.long)


measurment_tensor = pc_gt_lst[num+1,:][surface_indeces,:]
measurment_tensor = measurment_tensor.view(-1)
N_obs = measurment_tensor.shape[0]
alpha = 0.01
Cd = 0.01*torch.eye(N_obs,dtype=torch.float32)



#a = torch.FloatTensor(pc_lst[num,:])
#pc_lst_sample = torch.unsqueeze(a,0).cuda()
#latent_mu, latent_var =  model.encoder(pc_lst_sample)
#print(latent_var.shape)
#print((0.5*latent_var).exp())
#covarinace = torch.diag(torch.squeeze(latent_var,0))

Latent_ensambels = encoder_function (model,N_ensambels, pc_lst[num,:]) 
out_put_surface_nodes_prediction =  measurment_function (model, Latent_ensambels, surface_indeces)
Ensambel_predictions = out_put_surface_nodes_prediction.view(out_put_surface_nodes_prediction.size(0),-1)



esmda_model = esmda.ESMDA( Na, N_ensambels, N_obs, Latent_ensambels.t(), Ensambel_predictions.t(), Cd, alpha,measurment_function,model,surface_indeces)
a , b = esmda_model.assimilate (measurment_tensor)


gt = measurment_tensor.cuda()

print(torch.norm(torch.mean(b,1)-gt))
print(torch.norm(torch.mean(Ensambel_predictions,0)-gt))

final_latent = torch.unsqueeze(torch.mean(a,1),0)
first_latent = torch.unsqueeze(torch.mean(Latent_ensambels,0),0)


final_refined_mesh =  model.decoder(final_latent)
first_unrefined_mesh =  model.decoder(first_latent)


#print(pc_gt_lst[num,:].cuda()-torch.squeeze(first_unrefined_mesh,0))
#print(pc_gt_lst[num,:].cuda()-torch.squeeze(final_refined_mesh,0))

#print("before refinment",torch.norm(pc_gt_lst[num,:].cuda()-torch.squeeze(first_unrefined_mesh,0),dim=1))
#print("after refinment",torch.norm(pc_gt_lst[num,:].cuda()-torch.squeeze(final_refined_mesh,0),dim=1))

print("before refinment",torch.mean(torch.norm(pc_gt_lst[num+1,:].cuda()-torch.squeeze(first_unrefined_mesh,0),dim=1)))
print("after refinment",torch.mean(torch.norm(pc_gt_lst[num+1,:].cuda()-torch.squeeze(final_refined_mesh,0),dim=1)))

# # Save Assimilated out_put 

# pc_out = np.array(torch.squeeze(final_refined_mesh,0).data.tolist())
# pc_out0 = np.array(torch.squeeze(first_unrefined_mesh,0).data.tolist())
# pc_outgt = np.array(pc_gt_lst[num+1,:].data.tolist())
# template_plydata = PlyData.read(param.template_ply_fn)

# if not os.path.exists(out_ply_folder):
#     os.makedirs(out_ply_folder)

# Dataloader.save_pc_into_ply(template_plydata, pc_out, out_ply_folder+"f_out3.ply")
# Dataloader.save_pc_into_ply(template_plydata, pc_out0, out_ply_folder+"first3.ply")
# Dataloader.save_pc_into_ply(template_plydata, pc_outgt, out_ply_folder+"gt3.ply")
