import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from datafree.hooks import FeatureHook

_BN_NODES = {
    'resnet34': ['layer4.2.bn2', 'layer3.5.bn2', 'layer2.3.bn2', 'layer1.2.bn2'],
    'resnet50_imagenet': ['layer4.2.bn3','layer3.5.bn3','layer2.3.bn3', 'layer1.2.bn2', 'bn1'],
    'wrn40_2': ['bn1', 'block3.layer.0.bn1', 'block2.layer.0.bn1', 'block1.layer.0.bn1'],
    'vgg11': ['block4.4', 'block2.4', 'block1.1', 'block0.1']
}


def get_cfg(teacher, img_size, net_name):
    bn_nodes = _BN_NODES[net_name]
    hooks = {}
    for name, m in teacher.named_modules():
        name = name.replace('module.', '')
        if name in bn_nodes:
            hooks[name] = FeatureHook(m)
    test_data = torch.randn(1,img_size[0],img_size[1],img_size[2]).cuda()
    with torch.no_grad():
        teacher(test_data)
    
    cfg_list = []
    for i, node in enumerate(bn_nodes):
        _, bn_dim, size, _ = hooks[node].shape
        if i!=0:
            up_fac = size//pre_size
            # is_up = size!=pre_size
            cfg_list.append((pre_dim, bn_dim, up_fac))
        else:
            init_size = size
        pre_size = size
        pre_dim = bn_dim
    up_fac = img_size[-1]//pre_size
    # is_up = img_size[-1]!=pre_size
    cfg_list.append((bn_dim, img_size[0], up_fac)) 
    for h in hooks.values():
        h.remove()
    
    return cfg_list, init_size, bn_nodes


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return torch.flatten(x, 1)


class NLGenerator(nn.Module):
    def __init__(self, ngf=64, img_size=32, nc=3, nl=100, label_emb=None, le_emb_size=256, le_size=512, sbz=200):
        super(NLGenerator, self).__init__()
        self.params = (ngf, img_size, nc, nl, label_emb, le_emb_size, le_size, sbz)
        self.le_emb_size = le_emb_size
        # self.label_emb = label_emb
        self.label_emb = nn.parameter.Parameter(label_emb, requires_grad=False) # N_Cx512
        self.init_size = img_size // 4
        self.le_size = le_size
        self.nl = nl
        self.nle = int(np.ceil(sbz/nl))
        self.sbz = sbz

        self.n1 = nn.BatchNorm1d(le_size)
        self.sig1 = nn.Sigmoid()
        self.le1 = nn.ModuleList([nn.Linear(le_size, le_emb_size) for i in range(self.nle)])
        self.l1 = nn.Sequential(nn.Linear(le_emb_size, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.dr1 = nn.Dropout(p=0.25)
        self.le_sig = nn.Sigmoid()

    def re_init_le(self):
        for i in range(self.nle):
            nn.init.normal_(self.le1[i].weight, mean=0, std=1)
            nn.init.constant_(self.le1[i].bias, 0)

    def forward(self, targets=None):
        le = self.label_emb[targets]
        # le = self.sig1(le)
        le = self.n1(le)
        v = None
        for i in range(self.nle):
            if (i+1)*self.nl > le.shape[0]:
                sle = le[i*self.nl:]
            else:
                sle = le[i*self.nl:(i+1)*self.nl]
            sv = self.le1[i](sle)
            if v is None:
                v = sv
            else:
                v = torch.cat((v, sv))

        out = self.l1(v)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def reinit(self):
        return NLGenerator(self.params[0], self.params[1], self.params[2], self.params[3], self.params[4],
                             self.params[5], self.params[6], self.params[7]).cuda()


class NLDeepGenerator(nn.Module):
    def __init__(self, ngf=64, img_size=224, nc=3, nl=100, label_emb=None, le_emb_size=256, le_size=512, sbz=200):
        super(NLDeepGenerator, self).__init__()
        self.params = (ngf, img_size, nc, nl, label_emb, le_emb_size, le_size, sbz)
        self.le_emb_size = le_emb_size
        # self.label_emb = label_emb
        self.label_emb = nn.parameter.Parameter(label_emb, requires_grad=False) # N_Cx512
        self.init_size = img_size // 16
        self.le_size = le_size
        self.nl = nl
        self.nle = int(np.ceil(sbz/nl))
        self.sbz = sbz

        self.n1 = nn.BatchNorm1d(le_size)
        self.sig1 = nn.Sigmoid()
        self.le1 = nn.ModuleList([nn.Linear(le_size, le_emb_size) for i in range(self.nle)])
        self.l1 = nn.Sequential(nn.Linear(le_emb_size, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            #nn.Conv2d(nz, ngf, 3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(ngf),
            #nn.LeakyReLU(0.2, inplace=True),
            # 7x7

            #nn.Upsample(scale_factor=2),
            nn.Conv2d(2*ngf, 2*ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2*ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 14x14

            nn.Upsample(scale_factor=2),
            nn.Conv2d(2*ngf, 2*ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2*ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 28x28

            nn.Upsample(scale_factor=2),
            nn.Conv2d(2*ngf, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 56x56

            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 112 x 112

            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 224 x 224

            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def re_init_le(self):
        for i in range(self.nle):
            nn.init.normal_(self.le1[i].weight, mean=0, std=1)
            nn.init.constant_(self.le1[i].bias, 0)

    def forward(self, targets=None):
        le = self.label_emb[targets]
        # le = self.sig1(le)
        le = self.n1(le)
        v = None
        for i in range(self.nle):
            if (i+1)*self.nl > le.shape[0]:
                sle = le[i*self.nl:]
            else:
                sle = le[i*self.nl:(i+1)*self.nl]
            sv = self.le1[i](sle)
            if v is None:
                v = sv
            else:
                v = torch.cat((v, sv))

        out = self.l1(v)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def reinit(self):
        return NLDeepGenerator(self.params[0], self.params[1], self.params[2], self.params[3], self.params[4],
                               self.params[5], self.params[6], self.params[7]).cuda()


class InvGenerator(nn.Module):
    def __init__(self, ngf=512, img_size=32, nc=3, nl=100, label_emb=None, le_emb_size=256, le_size=512, sbz=200):
        super(InvGenerator, self).__init__()
        self.params = (ngf, img_size, nc, nl, label_emb, le_emb_size, le_size, sbz)
        self.le_emb_size = le_emb_size
        self.init_size = img_size // 8
        self.le_size = le_size
        self.nl = nl
        self.nle = int(np.ceil(sbz/nl))
        self.sbz = sbz
        self.label_emb = nn.parameter.Parameter(label_emb, requires_grad=False) # N_Cx512
        
        self.n1 = nn.BatchNorm1d(le_size)
        self.sig1 = nn.Sigmoid()
        self.le1 = nn.ModuleList([nn.Linear(le_size, le_emb_size) for i in range(self.nle)])
        self.l1 = nn.Sequential(nn.Linear(le_emb_size, ngf * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, ngf//2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf//2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf//2, ngf//4, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf//4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf//4, ngf//8, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf//8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ngf//8, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.dr1 = nn.Dropout(p=0.25)
        self.le_sig = nn.Sigmoid()

    def re_init_le(self):
        for i in range(self.nle):
            nn.init.normal_(self.le1[i].weight, mean=0, std=1)
            nn.init.constant_(self.le1[i].bias, 0)

    def forward(self, targets):
        le = self.label_emb[targets]
        le = self.n1(le)
        v = None
        for i in range(self.nle):
            if (i+1)*self.nl > le.shape[0]:
                sle = le[i*self.nl:]
            else:
                sle = le[i*self.nl:(i+1)*self.nl]
            sv = self.le1[i](sle)
            if v is None:
                v = sv
            else:
                v = torch.cat((v, sv))
        
        out = self.l1(v)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img, out

    def reinit(self):
        return InvGenerator(self.params[0], self.params[1], self.params[2], self.params[3], self.params[4],
                             self.params[5], self.params[6], self.params[7]).cuda()


class CoupledGenerator(nn.Module):
    def __init__(self, cfg_list, n_classes=100, init_size=4, le_emb_size=256, label_emb=None, le_size=512, sbz=200):
        super(CoupledGenerator, self).__init__()
        self.params = (cfg_list, n_classes, init_size, le_emb_size, label_emb, le_size, sbz)
        self.n_classes = n_classes
        self.init_size = init_size
        self.nle = int(np.ceil(sbz/n_classes))
        self.label_emb = nn.parameter.Parameter(label_emb, requires_grad=False) # N_Cx512
        self.syn_dict = {}
        
        # Define Generator
        self.bn1 = nn.BatchNorm1d(le_size)
        self.le1 = nn.ModuleList([nn.Linear(le_size, le_emb_size) for i in range(self.nle)])
        self.l1 = nn.Sequential(nn.Linear(le_emb_size, cfg_list[0][0]*self.init_size**2))
        
        self.bn2 = nn.BatchNorm2d(cfg_list[0][0])
        self.gen_list = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=up_fac) if up_fac!=1 else nn.Identity(),
                nn.Conv2d(in_d, out_d, 3, stride=1, padding=1) if i==len(cfg_list)-1 \
                    else nn.Conv2d(in_d, out_d, 3, stride=1, padding=1, bias=False),
                nn.Identity() if i==len(cfg_list)-1 else nn.BatchNorm2d(out_d),
                nn.Sigmoid() if i==len(cfg_list)-1 else nn.LeakyReLU(0.2, inplace=True)
                ) for i, (in_d, out_d, up_fac) in enumerate(cfg_list)])
        
    def re_init_le(self):
        for i in range(self.nle):
            nn.init.normal_(self.le1[i].weight, mean=0, std=1)
            nn.init.constant_(self.le1[i].bias, 0)

    def reinit(self):
        return CoupledGenerator(self.params[0], self.params[1], self.params[2], self.params[3],
                                self.params[4], self.params[5], self.params[6]).cuda()
        
    def forward(self, targets):
        dev_id = targets.device.index
        if dev_id not in self.syn_dict.keys():
            self.syn_dict[dev_id] = []
             
        le = self.label_emb[targets]
        le = self.bn1(le)
        v = None
        for i in range(self.nle):
            if (i+1)*self.n_classes > le.shape[0]:
                sle = le[i*self.n_classes:]
            else:
                sle = le[i*self.n_classes:(i+1)*self.n_classes]
            sv = self.le1[i](sle)
            if v is None:
                v = sv
            else:
                v = torch.cat((v, sv))
        out = self.l1(v)
        x = out.view(out.shape[0], -1, self.init_size, self.init_size)
        x = self.bn2(x)
        
        for i, gen in enumerate(self.gen_list):
            self.syn_dict[dev_id].append(x)
            x = gen(x)
        
        return x
    
    def syn_img_list(self):
        img_list = []
        for i in range(len(self.gen_list)):
            val = []
            for j in range(len(self.syn_dict)):
                val.append(self.syn_dict[j][i].cuda())
            img_list.append(torch.cat(val))
        self.syn_dict = {}
        return img_list
    
    
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3, nl=100):
        super(Generator, self).__init__()
        self.params = (nz, ngf, img_size, nc, nl)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z, y=None):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    # return a copy of its own
    def clone(self):
        clone = Generator(self.params[0], self.params[1], self.params[2], self.params[3], self.params[4])
        clone.load_state_dict(self.state_dict())
        return clone.cuda()


class DeepGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=224, nc=3):
        super(DeepGenerator, self).__init__()
        self.params = (nz, ngf, img_size, nc)
        self.init_size = img_size // 32
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            #nn.Conv2d(nz, ngf, 3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(ngf),
            #nn.LeakyReLU(0.2, inplace=True),
            # 7x7

            #nn.Upsample(scale_factor=2),
            nn.Conv2d(nz, 2*ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2*ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 14x14

            nn.Upsample(scale_factor=2),
            nn.Conv2d(2*ngf, 2*ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2*ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 28x28

            nn.Upsample(scale_factor=2),
            nn.Conv2d(2*ngf, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 56x56

            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 112 x 112

            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 224 x 224

            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),  
        )

    def forward(self, z):
        img = self.conv_blocks(z)
        return img

    # return a copy of its own
    def clone(self, copy_params=True):
        clone = DeepGenerator(self.params[0], self.params[1], self.params[2], self.params[3])
        if copy_params:
            clone.load_state_dict(self.state_dict())
        return clone.cuda()

        
class DCGAN_Generator(nn.Module):
    """ Generator from DCGAN: https://arxiv.org/abs/1511.06434
    """

    def __init__(self, ngf=64, img_size=224, nc=3, nl=100, label_emb=None, le_emb_size=256, le_size=512, sbz=200, slope=0.2):
        super(DCGAN_Generator, self).__init__()
        self.params = (ngf, img_size, nc, nl, label_emb, le_emb_size, le_size, sbz, slope)
        if isinstance(img_size, (list, tuple)):
            self.init_size = (img_size[0] // 16, img_size[1] // 16)
        else:
            self.init_size = (img_size // 16, img_size // 16)

        self.le_emb_size = le_emb_size
        self.label_emb = label_emb
        self.le_size = le_size
        self.nl = nl
        self.nle = int(np.ceil(sbz / nl))
        self.sbz = sbz

        self.n1 = nn.BatchNorm1d(le_size)
        self.sig1 = nn.Sigmoid()
        self.le1 = nn.ModuleList([nn.Linear(le_size, le_emb_size) for i in range(self.nle)])

        self.project = nn.Sequential(
            Flatten(),
            nn.Linear(le_emb_size, ngf * 8 * self.init_size[0] * self.init_size[1]),
        )

        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf * 8),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(slope, inplace=True),
            # 2x

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(slope, inplace=True),
            # 4x

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 8x

            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 16x

            nn.Conv2d(ngf, nc, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, targets=None):
        le = self.label_emb[targets]
        # le = self.sig1(le)
        le = self.n1(le)
        v = None
        for i in range(self.nle):
            if (i + 1) * self.nl > le.shape[0]:
                sle = le[i * self.nl:]
            else:
                sle = le[i * self.nl:(i + 1) * self.nl]
            sv = self.le1[i](sle)
            if v is None:
                v = sv
            else:
                v = torch.cat((v, sv))

        proj = self.project(v)
        proj = proj.view(proj.shape[0], -1, self.init_size[0], self.init_size[1])
        output = self.main(proj)
        return output

    def re_init_le(self):
        for i in range(self.nle):
            nn.init.normal_(self.le1[i].weight, mean=0, std=1)
            nn.init.constant_(self.le1[i].bias, 0)
    # return a copy of its own
    def clone(self, copy_params=True):
        clone = DeepGenerator(self.params[0], self.params[1], self.params[2], self.params[3])
        if copy_params:
            clone.load_state_dict(self.state_dict())
        return clone.cuda()

    def reinit(self):
        return DCGAN_Generator(self.params[0], self.params[1], self.params[2], self.params[3], self.params[4],
                               self.params[5], self.params[6], self.params[7], self.params[8]).cuda()


class DCGAN_CondGenerator(nn.Module):
    """ Generator from DCGAN: https://arxiv.org/abs/1511.06434
    """
    def __init__(self, num_classes,  nz=100, n_emb=50, ngf=64, nc=3, img_size=64, slope=0.2):
        super(DCGAN_CondGenerator, self).__init__()
        self.nz = nz
        self.emb = nn.Embedding(num_classes, n_emb)
        if isinstance(img_size, (list, tuple)):
            self.init_size = ( img_size[0]//16, img_size[1]//16 )
        else:    
            self.init_size = ( img_size // 16, img_size // 16)

        self.project = nn.Sequential(
            Flatten(),
            nn.Linear(nz+n_emb, ngf*8*self.init_size[0]*self.init_size[1]),
        )

        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf*8),
            
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(slope, inplace=True),
            # 2x

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(slope, inplace=True),
            # 4x
            
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 8x

            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 16x

            nn.Conv2d(ngf, nc, 3, 1,1),
            #nn.Tanh(),
            nn.Sigmoid()
        )

    def forward(self, z, y):
        y = self.emb(y)
        z = torch.cat([z, y], dim=1)
        proj = self.project(z)
        proj = proj.view(proj.shape[0], -1, self.init_size[0], self.init_size[1])
        output = self.main(proj)
        return output


class Discriminator(nn.Module):
    def __init__(self, nc=3, img_size=32):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(nc, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


class DCGAN_Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(DCGAN_Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)