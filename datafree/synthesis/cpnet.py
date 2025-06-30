import time
import math
import random
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook, FeatureHook
from datafree.criterions import custom_cross_entropy, kldiv
from datafree.utils import ImagePool, DataIter, AverageMeter
from torchvision import transforms
from kornia import augmentation


class CoupledSynthesizer(BaseSynthesis):
    def __init__(
        self, teacher, student, generator, num_classes, img_size, bn_nodes=[],
        g_steps=100, lr_g=0.1, adv=0.0, bn=1, oh=1, dist=0.5, warmup=10,
        synthesis_batch_size=128, sample_batch_size=128, cr_loop=1,
        bn_mmt=0, bnt=30, oht=1.5, g_life=50, g_loops=1, gwp_loops=10, normalizer=None,
        save_dir='run/cpnet', transform=None, dataset="cifar10", device='cpu'):
        super(CoupledSynthesizer, self).__init__(teacher, student)
        self.save_dir = save_dir
        self.img_size = img_size
        self.g_steps = g_steps
        self.bn_nodes = bn_nodes

        self.lr_g = lr_g
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.dist = dist
        self.bn_mmt = bn_mmt
        self.smoothing = 0.1
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.smoothing)

        self.device = device
        self.data_iter = None
        self.transform = transform
        self.generator = generator
        self.num_classes = num_classes
        self.sample_batch_size = sample_batch_size
        self.data_pool = ImagePool(root=self.save_dir)
        self.synthesis_batch_size = int(synthesis_batch_size/cr_loop)

        self.ep = 0
        self.ep_start = warmup
        self.g_life = g_life
        self.bnt = bnt
        self.oht = oht
        self.g_loops = g_loops
        self.gwp_loops = gwp_loops
        self.dataset = dataset
        self.label_list = torch.LongTensor([i for i in range(self.num_classes)])

        self.hooks = {}
        for name, m in teacher.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks[name.replace('module.', '')]=DeepInversionHook(m, self.bn_mmt)  
        
        if hasattr(teacher, 'module'):
            raw_teacher = teacher.module
        else:
            raw_teacher = teacher
        
        if hasattr(raw_teacher, 'linear'):
            self.fc_layer = raw_teacher.linear
        elif hasattr(raw_teacher, 'fc'):
            self.fc_layer = raw_teacher.fc
        elif hasattr(raw_teacher, 'classifier'):
            self.fc_layer = raw_teacher.classifier
        self.wpinv = compute_pinv(self.fc_layer.weight.T.detach(), 1e-1)
        
        if dataset == "imagenet":# or dataset == "tiny_imagenet":
            self.aug = transforms.Compose([
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                normalizer,
            ])
        else:
            self.aug = transforms.Compose([
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
                augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.4),
                normalizer,
            ])

    def synthesize(self, targets=None, use_cutmix=True):
        avg_loss = AverageMeter()
        avg_loss_oh = AverageMeter()
        avg_loss_bn = AverageMeter()
        avg_loss_adv = AverageMeter()
        avg_loss_xn = AverageMeter()
        avg_loss_dis = AverageMeter() 
        
        start = time.time()
        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6
        best_oh = 1e6
        if use_cutmix:
            cutmix = augmentation.RandomCutMixV2(p=0.5,data_keys=["input", "class"])
        
        if (self.ep - self.ep_start) % self.g_life == 0 or self.ep % self.g_life == 0:
            if hasattr(self.generator, 'module'):
                self.generator = self.generator.module.reinit()
                self.generator = torch.nn.DataParallel(self.generator)
            else:
                self.generator = self.generator.reinit()
                
        if self.ep < self.ep_start:
            g_loops = self.gwp_loops
        else:
            g_loops = self.g_loops
        self.ep += 1
        bi_list = []
        if g_loops == 0:
            return None, 0, 0, 0
        if self.dataset == "imagenet":
            idx = torch.randperm(self.label_list.shape[0])
            self.label_list = self.label_list[idx]
        for gs in range(g_loops):
            best_inputs = None
            if self.dataset == "imagenet":
                targets, ys = self.generate_ys_in(cr=0.0, i=gs)
            else:
                targets, ys = self.generate_ys(cr=0.0)
            ys = ys.to(self.device)
            targets = targets.to(self.device)
            
            if hasattr(self.generator, 'module'):
                self.generator.module.re_init_le()
                optimizer = torch.optim.Adam([
                    {'params': [p for p in self.generator.module.parameters() if p.requires_grad]},
                ], lr=self.lr_g, betas=[0.5, 0.999])
            else:
                self.generator.re_init_le()
                optimizer = torch.optim.Adam([
                    {'params': [p for p in self.generator.parameters() if p.requires_grad]},
                ], lr=self.lr_g, betas=[0.5, 0.999])

            feature_list = []
            for it in range(self.g_steps):
                inputs = self.generator(targets)
                if hasattr(self.generator, 'module'):
                    img_list = self.generator.module.syn_img_list()
                else:
                    img_list = self.generator.syn_img_list()
                if self.dataset == "imagenet":
                    inputs = self.jitter_and_flip(inputs)
                inputs_aug = self.aug(inputs)
                if use_cutmix:
                    inputs_aug, targets_mix = cutmix(inputs_aug, targets)
                
                t_out, t_feat = self.teacher(inputs_aug,return_features=True)
                [h.syn_values() for h in self.hooks.values()]
                loss_bn = sum([h.r_feature for k, h in self.hooks.items() if hasattr(h, 'r_feature')])
                if use_cutmix:
                    loss_oh = loss_mixup(targets_mix[0].detach(), t_out)
                else:
                    loss_oh = custom_cross_entropy(t_out, ys.detach())
                
                feature_list = [h.s_feature.detach() for k,h in self.hooks.items() if k in self.bn_nodes]
                img_list.reverse()
                loss_xn = sum([F.mse_loss(img, feat) for img, feat in zip(img_list, feature_list)])
                
                if self.adv > 0 and (self.ep > self.ep_start):
                    s_out = self.student(inputs_aug)
                    mask = (s_out.max(1)[1] == t_out.max(1)[1]).float()
                    loss_adv = -(kldiv(t_out, s_out, reduction='none').sum(1) * mask).mean()  # decision adversarial distillation
                    loss_align = self.compute_dis(t_feat, ys.detach())  # adversar feature distance
                    loss_adv += loss_align
                else:
                    loss_adv = loss_bn.new_zeros(1)
                
                loss = self.bn*loss_bn + self.oh*loss_oh + self.adv*loss_adv + loss_xn
                
                if loss_oh.item()<best_oh:
                    best_oh = loss_oh.item()
                
                avg_loss.update(loss.item())
                avg_loss_adv.update(loss_adv.item())
                avg_loss_bn.update(loss_bn.item())
                avg_loss_oh.update(loss_oh.item())
                avg_loss_xn.update(loss_xn.item())
                avg_loss_dis.update(loss_dis.item())

                with torch.no_grad():
                    if best_cost > loss.item() or best_inputs is None:
                        best_cost = loss.item()
                        best_inputs = inputs.data

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.bn_mmt != 0:
                for h in self.hooks.values():
                    h.update_mmt()

            self.student.train()
            end = time.time()
            self.data_pool.add(best_inputs)
            bi_list.append(best_inputs)

            dst = self.data_pool.get_dataset(transform=self.transform)
            loader = torch.utils.data.DataLoader(dst, batch_size=self.sample_batch_size, shuffle=True, num_workers=4, pin_memory=True)
            self.data_iter = DataIter(loader)
            print('Synthesising [{}/{}] Loss:{:.3f} Loss_oh:{:.3f} Loss_bn:{:.3f} Loss_adv:{:.3f} Loss_dis:{:.3f} Loss_xn {:.3f}'.format(
                    gs+1, g_loops, avg_loss.avg, avg_loss_oh.avg, avg_loss_bn.avg, avg_loss_adv.avg, avg_loss_dis.avg, avg_loss_xn.avg))
        return {"synthetic": bi_list}, end - start, best_cost, best_oh

    def sample(self):
        return self.data_iter.next()
    
    def jitter_and_flip(self, inputs_jit, lim=1. / 8., do_flip=True):
        lim_0, lim_1 = int(inputs_jit.shape[-2] * lim), int(inputs_jit.shape[-1] * lim)

        # apply random jitter offsets
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

        # Flipping
        flip = random.random() > 0.5
        if flip and do_flip:
            inputs_jit = torch.flip(inputs_jit, dims=(3,))
        return inputs_jit
    
    def generate_ys_in(self, cr=0.0, i=0):
        target = self.label_list[i*self.synthesis_batch_size:(i+1)*self.synthesis_batch_size]
        # target = torch.tensor([250, 230, 283, 282, 726, 895, 554, 555, 105, 107])

        ys = torch.zeros(self.synthesis_batch_size, self.num_classes)
        ys.fill_(cr / (self.num_classes - 1))
        ys.scatter_(1, target.data.unsqueeze(1), (1 - cr))

        return target, ys

    def generate_ys(self, cr=0.0):
        s = self.synthesis_batch_size // self.num_classes
        v = self.synthesis_batch_size % self.num_classes
        target = torch.randint(self.num_classes, (v,))
        for i in range(s):
            tmp_label = torch.tensor(range(0, self.num_classes))
            target = torch.cat((tmp_label, target))

        ys = torch.zeros(self.synthesis_batch_size, self.num_classes)
        ys.fill_(cr / (self.num_classes - 1))
        ys.scatter_(1, target.data.unsqueeze(1), (1 - cr))
        # print(target)

        return target, ys

    def compute_dis(self, feat, ys):
        bias = self.fc_layer.bias.detach() if hasattr(self.fc_layer, 'bias') else 0
        delta = math.log((1-self.smoothing)*(ys.shape[-1]-1)/self.smoothing)
        b = torch.ones_like(ys)@self.wpinv
        a = feat - ((ys*delta-bias)@self.wpinv)
        loss = F.cosine_similarity(a,b).abs()
        
        return loss.mean()


def compute_pinv(mat, threshold=1e-5):
    u, s, v = torch.linalg.svd(mat)
    s = torch.clamp(s, min=threshold)
    sigma_pinv = torch.zeros_like(mat).t()
    for i in range(min(u.shape[1], v.shape[0])):
        sigma_pinv[i, i] = 1 / s[i]
    mat_pinv = v.t() @ sigma_pinv @ u.t()
    return mat_pinv

def loss_mixup(y, logits):
    criterion = F.cross_entropy
    loss_a = criterion(logits, y[:, 0].long(), reduction='none')
    loss_b = criterion(logits, y[:, 1].long(), reduction='none')
    return ((1 - y[:, 2]) * loss_a + y[:, 2] * loss_b).mean()
