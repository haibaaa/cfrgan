#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os ; import sys 
os.chdir( os.path.split( os.path.realpath( sys.argv[0] ) )[0] ) 

import torch
import torch.backends.cudnn as cudnn

# Set CUDA device and configurations
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    cudnn.benchmark = True
    cudnn.deterministic = True
    # Enable TF32 for better performance on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Set memory allocator settings
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use up to 80% of GPU memory
    torch.cuda.empty_cache()

from mmRegressor.network.resnet50_task import *
from mmRegressor.preprocess_img import Preprocess
from mmRegressor.load_data import *
from mmRegressor.reconstruct_mesh import Reconstruction, Compute_rotation_matrix, _need_const, Projection_layer

import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2, glob
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    cameras, lighting,
    PointLights, HardPhongShader,
    RasterizationSettings,
    BlendParams,
    MeshRenderer, MeshRasterizer
)
from tqdm import tqdm
from tools.ops import erosion, SCDiffer, dilation, blur

# Retina Face
if os.path.exists('Pytorch_Retinaface'):
    from Pytorch_Retinaface.layers.functions.prior_box import PriorBox
    from Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
    from Pytorch_Retinaface.utils.box_utils import decode, decode_landm


class Aligner:
    def __init__(self, is_cuda=True, batch_size=1, render_size=224, test=True, model_path=None, back_white=False, cuda_id=0, det_net=None):
        self.is_cuda = is_cuda
        self.render_size = render_size
        self.cuda_id = cuda_id
        
        # Set CUDA device
        if is_cuda:
            torch.cuda.set_device(cuda_id)
            device = torch.device(f'cuda:{cuda_id}')
        else:
            device = torch.device('cpu')
            
        # Network, cfg
        if det_net is not None:
            self.det_net = det_net[0]
            self.det_cfg = det_net[1]

        # load models
        if model_path is None:
            print('Load pretrained weights')
        else:
            print(f'Load {model_path}')
        self.load_3dmm_models(model_path, test)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])
                
        self.argmax = lambda i, c: c[i]
        self.thresholding = nn.Threshold(0.3, 0.0)

        tri = self.face_model.tri
        tri = np.expand_dims(tri, 0)
        self.tri = torch.tensor(tri, dtype=torch.float32).repeat(batch_size, 1, 1)

        self.skin_mask = -1*self.face_model.skin_mask.unsqueeze(-1)
        
        if is_cuda:
            self.tri = self.tri.cuda(cuda_id)

        # Camera and renderer settings
        blend_params = BlendParams(background_color=(0.0,0.0,0.0))
        if back_white:
            blend_params= BlendParams(background_color=(1.0,1.0,1.0))

        self.R, self.T = look_at_view_transform(eye=[[0,0,10]], at=[[0,0,0]], up=[[0,1,0]], device=device)
        camera = cameras.FoVPerspectiveCameras(znear=0.01, zfar=50.0, aspect_ratio=1.0, fov=12.5936, R=self.R, T=self.T, device=device)
        lights = PointLights(ambient_color=[[1.0,1.0,1.0]], device=device, location=[[0.0,0.0,1e-5]])
        self.phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera,
                raster_settings=RasterizationSettings(
                    image_size=render_size,
                    blur_radius=0.0,
                    faces_per_pixel=1,
                    cull_backfaces=True
                )
            ),
            shader=HardPhongShader(cameras=camera, device=device, lights=lights, blend_params=blend_params)
        )

    def load_3dmm_models(self, model_path, test=True):
        # read face model
        self.face_model = BFM('mmRegressor/BFM/BFM_model_80.mat', self.cuda_id)

        # read standard landmarks for preprocessing images
        self.lm3D = self.face_model.load_lm3d("mmRegressor/BFM/similarity_Lm3D_all.mat")
        
        regressor = resnet50_use()
        if model_path is None:
            regressor.load_state_dict(torch.load("mmRegressor/network/th_model_params.pth"))
        else:
            regressor.load_state_dict(torch.load(model_path, map_location=f'cuda:{self.cuda_id}'))

        if test:
            regressor.eval()
        if self.is_cuda:
            regressor = regressor.cuda(self.cuda_id)
        if test:
            for param in regressor.parameters():
                param.requires_grad = False

        self.regressor = regressor

    def regress_3dmm(self, img):
        arr_coef = self.regressor(img)
        coef = torch.cat(arr_coef, 1)

        return coef

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--meta_path', type=str, required=True, help='5 landmarks file')
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--name', type=str, required=True, help='dataset name')
    args = parser.parse_args()

    # read face model
    face_model = BFM('mmRegressor/BFM/BFM_model_80.mat', -1)
    lm3D = face_model.load_lm3d("mmRegressor/BFM/similarity_Lm3D_all.mat")

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    fail_f = open(args.name + '_fail.txt', 'wt')
    with open(args.meta_path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if i%50==0:
            print('\r%.3f%%...' % (((i+1)/len(lines))*100), end='')

        splits = line.strip().split()
        fname = os.path.join(args.image_path, splits[0])
        if not os.path.exists(fname):
            fail_f.write(fname+'\n')
            continue
        
        lmk = list(map(float, splits[1:]))
        lmk = np.reshape(np.array(lmk), (5,2))
        img = Image.open(fname)
        try:
            img, _ = Preprocess(img, lmk, lm3D, 224)
        except ValueError:
            fail_f.write(fname+'\n')
            continue
        
        cv2.imwrite(os.path.join(args.save_path,splits[0]), img[0])
    
    fail_f.close()