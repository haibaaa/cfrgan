import numpy as np 
from PIL import Image
from scipy.io import loadmat,savemat
from array import array
import torch
import scipy.io as sio

# Set CUDA device and configurations
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    # Enable TF32 for better performance on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Set memory allocator settings
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use up to 80% of GPU memory
    torch.cuda.empty_cache()

# define facemodel for reconstruction
class BFM():
    def __init__(self, model_path, gpu_id=0):
        self.model = sio.loadmat(model_path)
        self.gpu_id = gpu_id
        
        # Move data to GPU
        if torch.cuda.is_available():
            self.meanshape = torch.from_numpy(self.model['meanshape']).cuda(gpu_id).squeeze(0)  # mean face shape
            self.idBase = torch.from_numpy(self.model['idBase']).cuda(gpu_id)  # identity basis
            self.exBase = torch.from_numpy(self.model['exBase']).type(torch.FloatTensor).cuda(gpu_id)  # expression basis
            self.meantex = torch.from_numpy(self.model['meantex']).cuda(gpu_id).squeeze(0)  # mean face texture
            self.texBase = torch.from_numpy(self.model['texBase']).cuda(gpu_id)  # texture basis
            self.skin_mask = torch.from_numpy(self.model['skinmask']).cuda(gpu_id)
        else:
            self.meanshape = torch.from_numpy(self.model['meanshape']).squeeze(0)  # mean face shape
            self.idBase = torch.from_numpy(self.model['idBase'])  # identity basis
            self.exBase = torch.from_numpy(self.model['exBase']).type(torch.FloatTensor)  # expression basis
            self.meantex = torch.from_numpy(self.model['meantex']).squeeze(0)  # mean face texture
            self.texBase = torch.from_numpy(self.model['texBase'])  # texture basis
            self.skin_mask = torch.from_numpy(self.model['skinmask'])
    
        self.tri = self.model['tri']
        self.keypoints = self.model['keypoints'].reshape(-1)
        self.point_buf = self.model['point_buf']
    

    # load landmarks for standard face, which is used for image preprocessing
    def load_lm3d(self, fsimilarity_Lm3D_all_mat):

        Lm3D = loadmat(fsimilarity_Lm3D_all_mat)
        Lm3D = Lm3D['lm']

        # calculate 5 facial landmarks using 68 landmarks
        lm_idx = np.array([31,37,40,43,46,49,55]) - 1
        Lm3D = np.stack([Lm3D[lm_idx[0],:],np.mean(Lm3D[lm_idx[[1,2]],:],0),np.mean(Lm3D[lm_idx[[3,4]],:],0),Lm3D[lm_idx[5],:],Lm3D[lm_idx[6],:]], axis = 0)
        Lm3D = Lm3D[[1,2,0,3,4],:]
        self.Lm3D = Lm3D
        return Lm3D


# load input images and corresponding 5 landmarks
def load_img(img_path,lm_path):

    image = Image.open(img_path)
    lm = np.loadtxt(lm_path)

    return image,lm

# save 3D face to obj file
def save_obj(path,v,f,c):
    with open(path,'w') as file:
        for i in range(len(v)):
            file.write('v %f %f %f %f %f %f\n'%(v[i,0],v[i,1],v[i,2],c[i,0],c[i,1],c[i,2]))

        file.write('\n')

        for i in range(len(f)):
            file.write('f %d %d %d\n'%(f[i,0],f[i,1],f[i,2]))

    file.close()

if __name__=='__main__':
    face_model = BFM("BFM/BFM_model_80.mat")
    # face_model.to_torch(is_torch=True, is_cuda=True)
    print(face_model.meantex.shape, face_model.meantex)
    print(face_model.texBase.shape, face_model.texBase)
    print(face_model.exBase.shape, face_model.exBase)