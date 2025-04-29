import argparse
import os, random, sys
import time, warnings
import PIL

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from model.networks import CFRNet, TwoScaleDiscriminator
from model.losses import PerceptualLoss, WGAN_DIV_Loss
from tools.datasets import CFRDataset
from tools.logger import CFRLogger
import random
from backbone import IR_SE_50

parser = argparse.ArgumentParser(description='Complete Face Recovery GAN')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the+ total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0002, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:29500', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true', default=True,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--n_critics', default=1, type=int)
# parser.add_argument('--schedule_iter', default=1500, type=int)
parser.add_argument('--log_iter', default=200, type=int)
parser.add_argument('--valid_iter', default=250, type=int)
parser.add_argument('--check_iter', default=5000, type=int)
parser.add_argument('--checkpoint_path', default='saved_models', type=str, help='path to save checkpoints')
parser.add_argument('--perceptual_model', default='resnet', type=str, help='vgg or resnet')
parser.add_argument('--img_path', default=None, required=True, type=str, help="original aligned image path")
parser.add_argument('--train_data_path', default=None, required=True, type=str, help="CFR-GAN train image data path")
parser.add_argument('--valid_data_path', default=None, required=True, type=str, help="CFR-GAN valid image data path")
parser.add_argument('--train_data_list', default=None, required=True, type=str, help="Train image list path")
parser.add_argument('--valid_data_list', default=None, required=True, type=str, help="Valid image list path")

logger = CFRLogger('logs')

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    # Create model
    print("=> creating model")
    model = CFRNet()
    discriminator = TwoScaleDiscriminator()
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            discriminator.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.gpu])
        else:
            model.cuda()
            discriminator.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            discriminator = torch.nn.parallel.DistributedDataParallel(discriminator)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        discriminator = discriminator.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            discriminator.features = torch.nn.DataParallel(discriminator.features)
            model.cuda()
            discriminator.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            discriminator = torch.nn.DataParallel(discriminator).cuda()

    criterion_per = PerceptualLoss(model_type=args.perceptual_model)
    criterion_l1 = torch.nn.L1Loss()
    criterion_l2 = torch.nn.MSELoss()
    criterion_wdiv = WGAN_DIV_Loss()
    criterion_cos = torch.nn.CosineSimilarity()

    if args.gpu is not None:
        criterion_per.cuda(args.gpu)
        
    nets = {}
    nets['G'] = model
    nets['D'] = discriminator
    
    optimizer_G = torch.optim.AdamW(model.parameters(), args.lr, betas=(0.5, 0.99))
    optimizer_D = torch.optim.AdamW(discriminator.parameters(), args.lr, betas=(0.5, 0.99))
    
    if args.resume and os.path.isfile(args.resume):
        dist.barrier()
        # map_location = {'cuda:%d' % 0: 'cuda:%d' % args.rank}
        map_location = 'cpu'
        print("Loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=map_location)
        args.start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['G_state_dict'])
        discriminator.load_state_dict(checkpoint['D_state_dict'])
        optimizer_G.load_state_dict(checkpoint['G_optimizer'])
        optimizer_D.load_state_dict(checkpoint['D_optimizer'])            
        for g in optimizer_G.param_groups:
            g['lr'] = checkpoint['G_lr']
        for g in optimizer_D.param_groups:
            g['lr'] = checkpoint['D_lr']

    nets['face'] = IR_SE_50([112,112])
    nets['face'].load_state_dict(torch.load('saved_models/face_res_50.pth'))
    nets['face'].eval()
    criterion_per.eval()
    
    cudnn.benchmark = True

    # Data loading code
    train_dataset = CFRDataset(args.img_path, args.train_data_path, args.train_data_list, img_size=224)
    valid_dataset = CFRDataset(args.img_path, args.valid_data_path, args.valid_data_list, test=True)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.workers, pin_memory=True)

    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=len(train_loader)*5, eta_min=1e-8)
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=len(train_loader)*5, eta_min=1e-8)
    schedulers = {}
    schedulers['G'] = scheduler_G
    schedulers['D'] = scheduler_D

    for epoch in range(args.start_epoch, args.epochs):    
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, valid_loader, nets, criterion_per, optimizer_G, optimizer_D, schedulers, epoch, args)

        # evaluate on validation set
        if torch.distributed.get_rank() == 0:
            # save checkpoint
            save_checkpoint({
                'epoch': epoch,
                'G_state_dict': model.state_dict(),
                'D_state_dict': discriminator.state_dict(),
                'G_optimizer': optimizer_G.state_dict(),
                'D_optimizer': optimizer_D.state_dict(),
                'G_lr' : scheduler_G.get_last_lr(),
                'D_lr' : scheduler_D.get_last_lr()
            }, args.checkpoint_path, epoch, len(train_loader)*(epoch+1))

            torch.save(model.state_dict(), args.checkpoint_path + '/CFRNet_G_ep{}.pth'.format(epoch+1))

def train(train_loader, valid_loader, nets, criterion_per, optimizer_G, optimizer_D, schedulers, epoch, args):
    val_iter = iter(valid_loader)
    
    # switch to train mode
    nets['G'].train()
    nets['D'].train()

    if args.perceptual_model=='vgg':
        Per_coef = 0.9
        GAN_coef = 1.0
        Occ_coef = 3.0
        Face_coef = 2.5
        Rec_coef = 0.3
    elif args.perceptual_model=='resnet':
        Per_coef = 3.0
        GAN_coef = 1.0
        Occ_coef = 2.0
        Face_coef = 1.0
        Rec_coef = 0.07
    else:
        raise ValueError('Model type should be one of resnet or vgg')
    

    mean = torch.FloatTensor([0.485, 0.456, 0.406]).cuda(args.gpu).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = torch.FloatTensor([0.229, 0.224, 0.225]).cuda(args.gpu).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    mean_f = torch.FloatTensor([0.5, 0.5, 0.5]).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std_f = torch.FloatTensor([0.5, 0.5, 0.5]).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    for i, (img, rot, gui, t_occ) in enumerate(train_loader):
        img = Variable(img.cuda(args.gpu, non_blocking=True), requires_grad=True)
        rot = rot.cuda(args.gpu, non_blocking=True)
        gui = gui.cuda(args.gpu, non_blocking=True)
        t_occ = t_occ.cuda(args.gpu, non_blocking=True)

        # -----------------------------------
        #  Train Discriminator (Real or Fake)
        # -----------------------------------

        optimizer_D.zero_grad()

        # compute output
        # normalize -1~1
        rot = (rot-0.5)*2
        gui = (gui-0.5)*2
        img = (img-0.5)*2

        output = nets['G'](rot, gui, wo_mask=True)
        
        # Real images
        real_validity = nets['D'](img)
        real_validity = torch.cat(real_validity, dim=1)

        # Generated images
        fake_validity = nets['D'](output)
        fake_validity = torch.cat(fake_validity, dim=1)

        # Total D loss
        loss_D = criterion_wdiv(real_validity, img, fake_validity, output)

        loss_D.backward()
        optimizer_D.step()

        # ------------------
        #  Train Generator
        # ------------------
        if i % args.n_critics == 0:
            optimizer_G.zero_grad()

            rot = rot.detach()
            gui = gui.detach()
            img = img.detach()
            
            # compute output
            output, occ_mask = nets['G'](rot, gui)
            
            # Adversarial loss
            fake_validity = nets['D'](output)
            fake_validity = torch.cat(fake_validity, dim=1)
            loss_GAN = -torch.mean(fake_validity)

            # Occlusion
            loss_occ_mask = criterion_l2(occ_mask, t_occ) * Occ_coef

            # Perceptual loss
            loss_per = torch.mean(criterion_per(((output*0.5+0.5)-mean)/std, ((img*0.5+0.5)-mean)/std)) * Per_coef

            # loss rec
            loss_rec = torch.sum(torch.abs(output - img), dim=1) * occ_mask
            loss_rec = torch.sum(loss_rec) / (torch.sum(occ_mask)+1e-7)
            loss_rec *= Rec_coef

            # Identity loss
            affined_o = F.interpolate(output[:,:,15:-40,15:-15], (112,112), mode='bilinear', align_corners=True)
            affined_i = F.interpolate(img[:,:,15:-40,15:-15], (112,112), mode='bilinear', align_corners=True)
            affined_o = affined_o * 0.5 + 0.5
            affined_i = affined_i * 0.5 + 0.5
            
            emb_o = nets['face']((affined_o-mean_f)/std_f)
            emb_i = nets['face']((affined_i-mean_f)/std_f)
            loss_id = torch.mean(1. - criterion_cos(emb_o, emb_i.detach())) * Face_coef

            # Total G loss
            loss_G = loss_GAN*GAN_coef + loss_per + loss_rec + loss_occ_mask + loss_id
            
            loss_G.backward(retain_graph=False)
            optimizer_G.step()

            # --------------
            #  Log Progress
            # --------------
            if torch.distributed.get_rank() == 0:
                total_iter = i+1+epoch*len(train_loader)
                # normalize 0~1
                rot = (rot/2)+0.5
                gui = (gui/2)+0.5
                img = (img/2)+0.5
                output = (output/2)+0.5

                if i % args.log_iter == 0:
                    rotated_grid = make_grid(rot, nrow=4, normalize=False)
                    guidance_grid = make_grid(gui, nrow=4, normalize=False)
                    output_grid = make_grid(output, nrow=4, normalize=False)
                    occ_grid = make_grid(occ_mask*rot, nrow=4, normalize=False)
                    target_grid = make_grid(img, nrow=4, normalize=True)
                    
                    logger.log_train_image(rotated_grid, guidance_grid, output_grid, occ_grid, target_grid, total_iter)

                if i % 2 == 0:
                    sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d]" % (epoch, args.epochs, i, len(train_loader)))
                    logger.log_training(loss_D.item(), loss_GAN.item(), loss_occ_mask.item(), loss_per.item(), loss_rec.item(), loss_id.item(), total_iter)
                
                schedulers['G'].step()

        if torch.distributed.get_rank() == 0: 
            schedulers['D'].step()

            if i!=0 and i % args.check_iter == 0:
                # save checkpoint
                save_checkpoint({
                    'epoch': epoch,
                    'G_state_dict': model.state_dict(),
                    'D_state_dict': discriminator.state_dict(),
                    'G_optimizer': optimizer_G.state_dict(),
                    'D_optimizer': optimizer_D.state_dict(),
                    'G_lr' : scheduler_G.get_last_lr(),
                    'D_lr' : scheduler_D.get_last_lr()
                }, args.checkpoint_path, epoch, len(train_loader)*(epoch+1))
            
            if (i+1) % args.valid_iter == 0:
                # test log
                try:
                    torch.cuda.empty_cache()
                    img, rotated, guidance = next(val_iter)
                except StopIteration:
                    val_iter = iter(valid_loader)
                    img, rotated, guidance = next(val_iter)
                
                with torch.no_grad():
                    img = Variable(img.cuda(args.gpu, non_blocking=True), requires_grad=True)
                    rotated = rotated.cuda(args.gpu, non_blocking=True)
                    guidance = guidance.cuda(args.gpu, non_blocking=True)
                    rotated = (rotated - 0.5) * 2
                    guidance = (guidance - 0.5) * 2

                    # make train data pair
                    output, occ_mask = nets['G'](rotated, guidance)
                    occ_mask = torch.round(occ_mask)
                    # normalize 0~1
                    rotated = (rotated / 2) + 0.5
                    guidance = (guidance / 2) + 0.5
                    output = (output / 2) + 0.5
                    
                    img_grid = make_grid(img, nrow=4, normalize=True)
                    rot_grid = make_grid(rotated, nrow=4, normalize=True)
                    gui_grid = make_grid(guidance, nrow=4, normalize=True)
                    output_grid = make_grid(output, nrow=4, normalize=True)
                    mask_grid = make_grid(occ_mask*rotated, nrow=4, normalize=True)
                    logger.log_test_image(img_grid, output_grid, rot_grid, gui_grid, mask_grid, total_iter)
                    
                    torch.cuda.empty_cache()                


def save_checkpoint(state, save_path, epoch, iteration=None):
    if iteration is not None:
        checkpoint_name = 'checkpoint_ep%d_it%d.pth' % (epoch+1, iteration)
    else:
        checkpoint_name = 'checkpoint_ep%d.pth' % (epoch+1)
    
    torch.save(state, os.path.join(save_path, checkpoint_name))
    

if __name__=='__main__':
    main()

