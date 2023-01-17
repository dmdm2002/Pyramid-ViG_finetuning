import os
import re

import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.Custom_DB_Loader import Loader
from utils.Option import param
from Official_code.pyramid_vig import pvig_b_224_gelu

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class test(param):
    def __init__(self):
        super(test, self).__init__()

        # output folder 생성
        os.makedirs(self.OUTPUT_CKP, exist_ok=True)
        os.makedirs(self.OUTPUT_LOG, exist_ok=True)

        # device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def init_weight(self, module):
        class_name = module.__class__.__name__

        if class_name.find("Conv2d") != -1 and class_name.find("EdgeConv2d") == -1 and class_name.find("DynConv2d") == -1:
            nn.init.normal_(module.weight.data, 0.0, 0.02)

        elif class_name.find("BatchNorm2d") != -1:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant(module.bias.data, 0.0)

    def run(self):
        print('--------------------------------------')
        print(f'[DEVICE] : {self.device}')
        print('--------------------------------------')

        # model = pvig_b_224_gelu().to(self.device)
        model = pvig_b_224_gelu(pretrained=True)
        model.prediction[4] = nn.Conv2d(1024, 10, kernel_size=(1, 1), stride=(1, 1))
        model.to(self.device)

        # ckp load
        ckp = torch.load(f'C:/Users/rlawj/WORK/SIDE_PROJECT/DACON/BLOCK_CLASSIFICATION/backup/try2/ckp_aug_fine_tuning/0.pth', map_location=self.device)
        model.load_state_dict(ckp['model_state_dict'])

        # define data transform
        data_transform = transforms.Compose([
            transforms.Resize(self.SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # Dataset Load
        te_dataset = Loader(self.ROOT, run_type='test', transform=data_transform, aug=False)
        te_loader = DataLoader(dataset=te_dataset, batch_size=1, shuffle=False)

        # Valid Iteration Loop
        print('--------------------------------------')
        print('Test Start!!')
        print('--------------------------------------')

        test_result_list = []
        model.eval()
        with torch.no_grad():
            for idx, (item, name) in enumerate(tqdm.tqdm(te_loader)):
                item = item.to(self.device)
                probs = model(item)

                probs = probs.cpu().detach().numpy()
                preds = probs > 0.5
                preds = preds.astype(int)

                ID = name[0].split('/')[-1]
                ID = re.compile(".jpg").sub("", ID)

                preds = preds.tolist()
                preds[0].insert(0, ID)

                test_result_list += preds

        print(test_result_list)
        submit = pd.DataFrame(test_result_list, columns=['id', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
        submit.to_csv('./VIG_PRETRAINED_CKP1_SUBMIT.csv', index=False)