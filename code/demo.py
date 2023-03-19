import os
import time
import argparse
import math
import re
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

from stylegan2.model import Generator

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, default="../ckpts/pretrained_model/stylegan2-ffhq-config-f.pt", help="pretrained StyleGAN2 weights")
    parser.add_argument("--image_size", type=int, default=1024, help="StyleGAN resolution")
    parser.add_argument("--truncation", type=float, default=0.7, help="used only for the initial latent vector")
    parser.add_argument("--depth", type=float, default=5, help="depth of ODE function")
    parser.add_argument("--style_dim", type=float, default=512, help="style_dim of ODE function")

    parser.add_argument("--latent_path", type=str, default="../data/example/inverted_code/all_latents.pth", help="path of inverted code")
    parser.add_argument('--data_size', type=int, default=10,  help="how many samples in total (this should not be large than the numbers of the latent codes defined in latent_path")
    parser.add_argument('--batch_time', type=int, default=5, help="how many time stamps are chosen in one batch")

    parser.add_argument("--niters", type=int, default=20000, help="number of optimization steps")
    parser.add_argument('--batch_size', type=int, default=1, help="how many samples are chosen in one batch")
    parser.add_argument("--lr", type=float, default=2e-4)

    parser.add_argument("--ckpt_save_path", type=str, default="../ckpts", help="the path to save model ckpt")
    parser.add_argument("--results_dir", type=str, default="../results")
    parser.add_argument('--test_save_freq', type=int, default=100, help="test save frequency")
    parser.add_argument('--training_save_freq', type=int, default=20, help="training save frequency")

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
    parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
    parser.add_argument('--adjoint', action='store_true')
    args = parser.parse_args()

    return args

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def load_generator(args):
    print("loading stylegan2 generator")
    g_ema = Generator(args.image_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=True)
    g_ema = g_ema.cuda()
    g_ema.eval()
    for param in g_ema.parameters():
        param.requires_grad = False
    return g_ema

class VideoLatentDataset():
    def __init__(self, latent_path):
        self.latents = torch.load(latent_path)
        self.t_length = len(self.latents)
        print("The loaded video contains {} frames.".format(self.t_length))
    
    def __getitem__(self, idx):
        return {
            "latents": self.latents[idx:idx+1],  # (T, 1, 18, 512)
            "t_steps": torch.Tensor([idx]).cuda()  # (T, )
        }
    
    def get_item(self, length=5):
        t_idx = np.random.choice(self.t_length, length, replace=False)
        # sort the selected frame idx
        t_idx = np.sort(t_idx)
        sampled_latents = self.latents[t_idx]
        return {
            "latents": sampled_latents.cuda(),  # (T, 1, 18, 512)
            "t_steps": torch.Tensor(t_idx).cuda()  # (T, )
        }
    
    def get_batch_items(self, length=5, batch_size=2):
        latents = []
        t_steps = []
        for i in range(batch_size):
            data = self.get_item(length)
            latents.append(data["latents"])
            t_steps.append(data["t_steps"])
        latents = torch.stack(latents, dim=0)
        t_steps = torch.stack(t_steps, dim=0)
        return {
            "latents": latents, # (B, T, 1, 18, 512)
            "t_steps": t_steps  # (B, T)
        }

    def __len__(self):
        """
        return the video length
        """
        return len(self.latents)


def _get_tensor_value(tensor):
    """Gets the value of a torch Tensor."""
    return tensor.cpu().detach().numpy()


class ODEfunc(nn.Module):
    def __init__(self, dim, depth=1):
        super().__init__()
        self.depth = depth
        layers = []
        for i in range(depth-1):
            layers += [nn.Linear(dim, dim), nn.LeakyReLU(0.2)]
        layers.append(nn.Linear(dim, dim))
        self.model = nn.Sequential(*layers)
        
    def forward(self, t, x, eps=1e-6):
        out = self.model(x)
        out = out / (torch.norm(out, dim=1, keepdim=True) + eps)
        return out

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

def train(args):
    '''
    optimize odefunc
    '''
    # define odefunc
    odefunc = ODEfunc(args.style_dim, depth=args.depth).cuda()
    print("ODEfunc: ", odefunc)
    g_reg_ratio = 4 / 5
    # optimizer = optim.RMSprop(odefunc.parameters(), lr=args.lr)
    optimizer = optim.Adam(odefunc.parameters(), 
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )
    
    # construct stylegan generator
    generator = load_generator(args)
    # construct dataset
    dataset = VideoLatentDataset(args.latent_path)

    end = time.time()

    time_meter = RunningAverageMeter(0.9)
    loss_meter = RunningAverageMeter(0.9)

    for itr in range(1, args.niters + 1):
        # batch_data = dataset.get_batch_items()
        single_data = dataset.get_item()
        pred_l = odeint(odefunc, single_data["latents"][0], single_data["t_steps"]).cuda() # (T, 1, 18, 512)

        # [T, 1, 18, 512] -> [T, 1, 18, 512] -> [T, 18, 512] 
        pred_l_flatten = pred_l.flatten(0, 1)
        true_l_flatten = single_data["latents"].flatten(0, 1)

        # generate images
        with torch.no_grad():
            true_i, _ = generator([true_l_flatten], input_is_latent=True, 
                randomize_noise=False, return_latents=False)

        pred_i = []
        for i in range(len(pred_l_flatten)):
            img_gen, _ = generator([pred_l_flatten[i:i+1]], input_is_latent=True, 
                randomize_noise=False, return_latents=False)
            pred_i.append(img_gen)
        pred_i = torch.cat(pred_i, dim=0)

        # resize images for loss calculation
        pred_i =  F.adaptive_avg_pool2d(pred_i, (256, 256))  # [T, 3, 256, 256]
        true_i = F.adaptive_avg_pool2d(true_i, (256, 256))

        if itr % args.training_save_freq  == 0:
            save_image(torch.cat([true_i.cpu().detach(), pred_i.cpu().detach()], dim=0), 
                os.path.join(args.results_dir, "training_{}.png".format(itr)), nrow=5, normalize=True, scale_each=True)

        loss = 0.0
        log_message = "lr: {} | Iter {:04d} | ".format(args.lr, itr)
        # latent reconstruction loss 
        loss_l = torch.mean((pred_l - single_data["latents"])**2)  #torch.mean(torch.abs(pred_l - batch_l))
        loss = loss + loss_l
        log_message += 'loss_latent: {:.3f}'.format(_get_tensor_value(loss_l))

        log_message += ' | total Loss: {:.6f}'.format(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_save_freq == 0:
            if not os.path.exists(args.ckpt_save_path):
                os.makedirs(args.ckpt_save_path, exist_ok=True)
            ckpt = odefunc.state_dict()
            torch.save(ckpt, "./ckpts/iters_{}.pth".format(itr))

            single_data = dataset.get_item(10)
            start_l0 = single_data["latents"][0]
            start_t0 = single_data["t_steps"][0]
            t_steps = torch.linspace(start_t0, 11, 12)
            pred_lt = odeint(odefunc, start_l0, t_steps)

            pred_lt_flatten = pred_lt.flatten(0, 1)  # (T, 18, 512)
            with torch.no_grad():
                img_gen_list = []
                for i in range(len(pred_lt_flatten)):
                    img_gen = generator([pred_lt_flatten[i:i+1]], input_is_latent=True, 
                    randomize_noise=False, return_latents=False)[0]
                    img_gen_list.append(F.adaptive_avg_pool2d(img_gen.detach().cpu(), (200, 200)))
            img_gen_list = torch.cat(img_gen_list, dim=0)
            save_image(img_gen_list, 
                os.path.join(args.results_dir, "test_{}_fake.png".format(itr)), nrow=6,
                normalize=True, scale_each=True)
        log_message += " | time: {}s".format(time.time() - end)
        print(log_message)
        end = time.time()


def test(args):
    # define odefunc
    odefunc = ODEfunc(args.style_dim, depth=args.depth).cuda()

    ckpt = torch.load("../ckpts/dynode_ckpt.pth")
    odefunc.load_state_dict(ckpt)

    print("ODEfunc: ", odefunc)

    # construct stylegan generator
    generator = load_generator(args)

    # construct dataset
    dataset = VideoLatentDataset(args.latent_path)
    data_start = dataset.get_item(length=1)
    # data_start = dataset[0]

    latent_l0 = data_start["latents"][0]
    start_t0 = data_start["t_steps"][0]
    t_steps = torch.linspace(start_t0, 11, 32)

    # produce interpolation results
    pred_lt = odeint(odefunc, latent_l0, t_steps) # (T, 1, 18, 512)

    pred_lt_flatten = pred_lt.flatten(0, 1)  # (T, 18, 512)
    with torch.no_grad():
        img_gen_list = []
        for i in range(len(pred_lt_flatten)):
            img_gen = generator([pred_lt_flatten[i:i+1]], input_is_latent=True, 
            randomize_noise=False, return_latents=False)[0]
            img_gen_list.append(F.adaptive_avg_pool2d(img_gen.detach().cpu(), (200, 200)))
    img_gen_list = torch.cat(img_gen_list, dim=0)

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    save_image(img_gen_list, os.path.join(args.results_dir, "interp_results.png"), nrow=8,
        normalize=True, scale_each=True)
    print("done!")

if __name__ == "__main__":
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

    if args.mode == 'train':
        train(args)
    else:
        test(args)