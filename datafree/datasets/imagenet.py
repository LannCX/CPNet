import os
from PIL import Image
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets, transforms as T


class ImagenetDataset(data.Dataset):
    def __init__(self, root, transform, split_root=''):
        self.root = root
        self.transform = transform
        split = root.split('/')[-1]
        cls_file = os.path.join(split_root, split, 'class_labels.txt')
        
        self.anno = []
        if cls_file:
            for l in open(cls_file, 'r').readlines():
                img_name, label = l.rstrip().split(',')
                img_name = img_name.split('/')[-1]
                self.anno.append({
                    'path':os.path.join(self.root, img_name),
                    'label': int(label),
                })
        self.is_test = None
    
    def __getitem__(self, index):
        path = self.anno[index]['path']
        label = self.anno[index]['label']
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img_tensor = self.transform(img)
        
        return img_tensor, label
    
    def __len__(self):
        return len(self.anno)
    
    

NORMALIZE_DICT = {
    'mnist':    dict( mean=(0.1307,),                std=(0.3081,) ),
    'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'cifar100': dict( mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) ),
    'imagenet': dict( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'tinyimagenet': dict( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    
    'cub200':   dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'stanford_dogs':   dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'stanford_cars':   dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'places365_32x32': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'places365_64x64': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'places365': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'svhn': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'tiny_imagenet': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'imagenet_32x32': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    
    # for semantic segmentation
    'camvid': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'nyuv2': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
}


if __name__=='__main__':
    import pdb
    from tqdm import tqdm
    tran = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT['imagenet']),
        ])
    imdb = ImagenetDataset(
        '/data1/jrh/V2L-Tokenizer-main/ImageNet1K/val',
        transform=tran,
        split_root='/data1/jrh/V2L-Tokenizer-main/imagenet_split'
        )
    for items in imdb:
        pdb.set_trace()
        print('wtf')
    