import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import resnet50
from model.networks import VGG19
# from utils import load_state_dict
from torch.autograd import Variable

# Set CUDA device and configurations
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    # Enable TF32 for better performance on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Set memory allocator settings
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use up to 80% of GPU memory
    torch.cuda.empty_cache()

class PerceptualLoss(nn.Module):
    def __init__(self, model_type='vgg'):
        super(PerceptualLoss, self).__init__()
        if model_type == 'vgg':
            from model.vgg import VGG19
            self.vgg = VGG19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()
        else:
            raise NotImplementedError('Model type [%s] is not implemented' % model_type)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, y_hat, y):
        return self.criterion(y_hat, y)


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.MSELoss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class ResLoss(nn.Module):
    def __init__(self):
        super(ResLoss, self).__init__()
        self.criterion = nn.L1Loss()
        resnet = resnet50(pretrained=True)

        self.B1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
        )
        self.B2 = resnet.layer2
        self.B3 = resnet.layer3

    def forward(self, _data, _target):
        data, target = self.B1(_data), self.B1(_target)
        B1_loss = torch.mean(torch.square(data - target), dim=[1,2,3])

        data, target = self.B2(data), self.B2(target)
        B2_loss = torch.mean(torch.square(data - target), dim=[1,2,3])

        data, target = self.B3(data), self.B3(target)
        B3_loss = torch.mean(torch.square(data - target), dim=[1,2,3])

        return B1_loss + B2_loss + B3_loss


class WGAN_DIV_Loss(nn.Module):
    def __init__(self, dim=1):
        super(WGAN_DIV_Loss, self).__init__()
        self.dim = dim

    def forward(self, real_val, real_img, fake_val, fake_img):
        device = real_val.device
        real_grad_out = Variable(torch.FloatTensor(real_val.size(0), self.dim).fill_(1.0), requires_grad=False).to(device)
        fake_grad_out = Variable(torch.FloatTensor(fake_val.size(0), self.dim).fill_(0.0), requires_grad=False).to(device)
        
        real_loss = F.binary_cross_entropy_with_logits(real_val, real_grad_out)
        fake_loss = F.binary_cross_entropy_with_logits(fake_val, fake_grad_out)
        
        return real_loss + fake_loss
    

