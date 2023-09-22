import os
import time

from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import matplotlib.pyplot as plt
import math
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import restore_checkpoint

from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import sampling
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling import (ReverseDiffusionPredictor,
                                  LangevinCorrector,
                                  EulerMaruyamaPredictor,
                                  AncestralSamplingPredictor,
                                  NoneCorrector,
                                  NonePredictor,
                                  AnnealedLangevinDynamics)
import datasets
from WRN import WideResNet

currenttime = time.time()

import argparse
parser = argparse.ArgumentParser(description='Demo of argparse')
parser.add_argument('--local-rank', type=int, default=-1)
# parser.add_argument('--local_rank', type=int, default=-1)

args = parser.parse_args()

torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group('nccl')


classifier = WideResNet(depth=28, widen_factor=10)
classifier.load_state_dict(torch.load("saved_weight/natural.pt", map_location=torch.device('cpu'))["state_dict"])

classifier = classifier.to(args.local_rank)
classifier = torch.nn.parallel.DistributedDataParallel(classifier,
                                        device_ids=[args.local_rank],
                                        output_device=args.local_rank,
                                        find_unused_parameters=False,
                                        broadcast_buffers=False)
# classifier.load_state_dict(torch.load("cotrain_0726_weight/checkpoint_classifier_88.pth", map_location=torch.device('cpu')))
classifier.eval()

from configs.vp import cifar10_ddpmpp_deep_continuous as configs
# ckpt_filename = "/media/data1/lkz_new/score_sde_pytorch_adv_0726/exp_0729_cotrain_genonly/checkpoint_83.pth"
# ckpt_filename = "/media/data1/lkz_new/score_sde_pytorch_adv_0726/score_sde_pytorch_adv_0726_data2/exp_0729_gentrainonly_20833/checkpoint_86.pth"
ckpt_filename = "saved_weight/checkpoint_8.pth"

config = configs.get_config()
sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
sampling_eps = 1e-3

img_num = 8
eot = 10
batch_size = img_num * eot

config.eval.batch_size = batch_size

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.get_model(config.model.name)(config).to(args.local_rank)
score_model = torch.nn.parallel.DistributedDataParallel(score_model,
                                        device_ids=[args.local_rank],
                                        output_device=args.local_rank,
                                        find_unused_parameters=False,
                                        broadcast_buffers=False)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, torch.device(args.local_rank))
ema.copy_to(score_model.parameters())

def image_grid(x):
    size = config.data.image_size
    channels = config.data.num_channels
    img = x.reshape(-1, size, size, channels)
    w = int(np.sqrt(img.shape[0]))
    img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
    return img

def show_samples(x):
    x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
    img = image_grid(x)
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.imshow(img)
    plt.show()

img_size = config.data.image_size
channels = config.data.num_channels
shape = (batch_size, channels, img_size, img_size)
predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
snr = 0.16 #@param {"type": "number"}
n_steps =  1#@param {"type": "integer"}
probability_flow = False #@param {"type": "boolean"}
sampling_fn = sampling.get_partial_pc_sampler(sde, shape, predictor, corrector,
                                      inverse_scaler, snr, tstart = 0.1, n_steps=n_steps,
                                      probability_flow=probability_flow,
                                      continuous=config.training.continuous,
                                      eps=sampling_eps, device=torch.device(args.local_rank))

test_data = torchvision.datasets.CIFAR10(root="./dataset",train=False, download=True, transform=transforms.ToTensor())
test_dataloader = DataLoader(dataset=test_data,batch_size=img_num, shuffle=False)

loss = torch.nn.CrossEntropyLoss()

correct = 0
total = 0
ws = torch.distributed.get_world_size()
group = torch.distributed.group.WORLD

score_model.eval()
classifier.eval()

for step, (b_x, b_y) in enumerate(test_dataloader):
    if step % ws != args.local_rank:
        continue

    if step >= 128:
        break

    b_x = b_x.to(args.local_rank)
    b_y = b_y.to(args.local_rank)
    purturb = torch.zeros_like(b_x)

    b_x = b_x.unsqueeze(1).repeat_interleave(eot, dim=1).view(-1, b_x.shape[1], b_x.shape[2], b_x.shape[3])
    b_y = b_y.unsqueeze(1).repeat_interleave(eot, dim=1).view(-1)

    for attack_epoch in range(20):
        purturb_ = purturb.unsqueeze(1).repeat_interleave(eot, dim=1).view(-1, b_x.shape[1], b_x.shape[2], b_x.shape[3])
        purturb_.requires_grad = True

        x = b_x + purturb_

        # show_samples(x)

        x, n = sampling_fn(score_model, scaler(x.to(args.local_rank)))
        result = classifier(x.to(args.local_rank))
        lossnum = loss(result, b_y)
        lossnum.backward()

        lossnum_ = lossnum.clone().detach()
        torch.distributed.barrier(group)
        torch.distributed.all_reduce(lossnum_, op=torch.distributed.ReduceOp.AVG)
        if args.local_rank == 0:
            print(lossnum_)

        grad = torch.mean(purturb_.grad.view(-1, eot, b_x.shape[1], b_x.shape[2], b_x.shape[3]), dim=1)

        purturb = purturb + grad.sign() * 2 / 255
        purturb = torch.clamp(purturb, -8 / 255, 8 / 255).detach()

        if args.local_rank == 0:
            print(f"avg purturb : {torch.mean(torch.abs(purturb))}")
            print(f"attack_epoch : {attack_epoch}")

    with torch.no_grad():

        purturb_ = purturb.unsqueeze(1).repeat_interleave(eot, dim=1).view(-1, b_x.shape[1], b_x.shape[2], b_x.shape[3])
        x = b_x + purturb_

        timeused = time.time()
        x, n = sampling_fn(score_model, scaler(x.to(args.local_rank)))
        timeused = time.time() - timeused

        result = classifier(x.to(args.local_rank))

    folder = "attack_samples_0829_timetestonly_allclean_1024"

    if not os.path.exists(folder):
        os.makedirs(folder)

    torch.save(b_x.view(-1, eot, b_x.shape[1], b_x.shape[2], b_x.shape[3])[:,0,:,:,:].cpu(), f"{folder}/ori_{step}_{args.local_rank}.pt")
    torch.save(purturb.cpu(), f"{folder}/purturb_{step}_{args.local_rank}.pt")
    torch.save(x.cpu(), f"{folder}/purified_{step}_{args.local_rank}.pt")
    torch.save(b_y.view(-1, eot)[:, 0].cpu(),
               f"{folder}/gt_{step}_{args.local_rank}.pt")

    correct += torch.sum(result.argmax(dim=1) == b_y)
    total += b_y.shape[0]

    correct_ = correct.clone().detach()
    total_ = torch.tensor(total).to(args.local_rank)

    torch.distributed.barrier(group)
    torch.distributed.all_reduce(correct_, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(total_, op=torch.distributed.ReduceOp.SUM)

    if args.local_rank == 0:
        print(f"correct : {correct_} \n total : {total_}")

        with open(folder + f"/log_{currenttime}.txt", "a") as log:
            log.write(f"correct : {correct_} \n total : {total_} \n")
            log.write(f"timeused : {timeused} \n")