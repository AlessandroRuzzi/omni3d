
import numpy as np
from imageio import imread
from PIL import Image

from termcolor import colored, cprint

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from .behave_dataset import BehaveDataset, BehaveImgDataset
from .intercap_dataset import IntercapDataset, IntercapImgDataset

from torchvision import datasets

from configs.paths import dataroot

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def CreateDataset(opt):
    val_dataset = None   

    # decide resolution later at model
        

    if opt.dataset_mode == 'behave':
        """ behave dataset. """
        train_dataset = BehaveDataset()
        test_dataset = BehaveDataset()
        val_dataset = BehaveDataset()
        train_dataset.initialize(opt, 'train', cat=opt.BEHAVE_train_cat)
        test_dataset.initialize(opt, 'test', cat=opt.BEHAVE_test_cat)
        val_dataset.initialize(opt, 'val', cat=opt.BEHAVE_val_cat)
        
    elif opt.dataset_mode == 'behave_img':
        """ behave dataset. """
        train_dataset = BehaveImgDataset()
        test_dataset = BehaveImgDataset()
        val_dataset = BehaveImgDataset()
        train_dataset.initialize(opt, 'train', cat=opt.BEHAVE_train_cat)
        test_dataset.initialize(opt, 'test', cat=opt.BEHAVE_test_cat)
        val_dataset.initialize(opt, 'val', cat=opt.BEHAVE_val_cat)
        
    elif opt.dataset_mode == 'intercap':
        """ intercap dataset. """
        train_dataset = IntercapDataset()
        test_dataset = IntercapDataset()
        val_dataset = IntercapDataset()
        train_dataset.initialize(opt, 'train', cat=opt.INTERCAP_train_cat)
        test_dataset.initialize(opt, 'test', cat=opt.INTERCAP_test_cat)
        val_dataset.initialize(opt, 'val', cat=opt.INTERCAP_val_cat)
        
    elif opt.dataset_mode == 'intercap_img':
        """ intercap dataset. """
        train_dataset = IntercapImgDataset()
        test_dataset = IntercapImgDataset()
        val_dataset = IntercapImgDataset()
        train_dataset.initialize(opt, 'train', cat=opt.INTERCAP_train_cat)
        test_dataset.initialize(opt, 'test', cat=opt.INTERCAP_test_cat)
        val_dataset.initialize(opt, 'val', cat=opt.INTERCAP_val_cat)


    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    cprint("[*] Dataset has been created: %s" % (train_dataset.name()), 'blue')
    return train_dataset, val_dataset, test_dataset
