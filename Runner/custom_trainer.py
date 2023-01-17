import gzip
import os
import tqdm
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.Custom_DB_Loader import Loader
from utils.Option import param
from torch.utils.tensorboard import SummaryWriter
from Model.PyramidViG import Pyramid_ViG
from Official_code.pyramid_vig import pvig_b_224_gelu

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class train(param):
    def __init__(self):
        super(train, self).__init__()

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

        # ckp load or model init
        if self.CKP_LOAD:
            ckp = torch.load(f'pvig_b_83.66.pth.tar', map_location=self.device)
            model.load_state_dict(ckp)
            epoch = 0

        else:
            model.apply(self.init_weight)
            epoch = 0

        # 마지막 conv 변경하여 모델 설정
        model.prediction[4] = nn.Conv2d(1024, 10, kernel_size=(1, 1), stride=(1, 1))
        model.to(self.device)

        # define data transform
        data_transform = transforms.Compose([
            transforms.Resize(self.SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # Dataset Load
        tr_dataset = Loader(self.ROOT, run_type='train', transform=data_transform, aug=True)
        valid_dataset = Loader(self.ROOT, run_type='valid', transform=data_transform, aug=True)

        # define loss function
        criterion = nn.BCEWithLogitsLoss()

        summary = SummaryWriter(self.OUTPUT_LOG)

        # define optimizer
        optim_adamw = optim.AdamW(list(model.parameters()), lr=self.LR, weight_decay=0.05)
        scheduler_cosin = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optim_adamw, T_0=20, T_mult=1, eta_min=1e-5)

        # save loss and acc
        score_list = []

        # Training Loop
        for ep in range(0, self.EPOCH):
            tr_loader = DataLoader(dataset=tr_dataset, batch_size=self.BATCHSZ, shuffle=True)
            valid_loader = DataLoader(dataset=valid_dataset, batch_size=self.BATCHSZ, shuffle=False)

            train_loss = []
            train_acc = []

            valid_loss = []
            valid_acc = []

            model.train()
            # Training Iteration Loop
            print('--------------------------------------')
            print('Training Start!!')
            print('--------------------------------------')
            for idx, (item, label) in enumerate(tqdm.tqdm(tr_loader, desc=f'NOW EPOCH [{ep} /{self.EPOCH}]')):
                item = item.to(self.device)
                label = label.to(self.device)

                probs = model(item)

                # Loss 계산
                loss_value = criterion(probs.view(-1).type(torch.FloatTensor), label.view(-1).type(torch.FloatTensor))

                # backward 및 update
                optim_adamw.zero_grad()
                loss_value.backward()
                optim_adamw.step()

                # loss and acc update
                probs = probs.cpu().detach().numpy()
                label = label.cpu().detach().numpy()
                preds = probs > 0.5
                batch_acc = (label == preds).mean()

                train_loss.append(loss_value.item())
                train_acc.append(batch_acc)

            scheduler_cosin.step(ep)
            train_acc_avg = np.mean(train_acc)
            train_loss_avg = np.mean(train_loss)

            print(f'TRAIN ACC : {train_acc_avg}')
            print(f'TRAIN LOSS : {train_loss_avg}')

            # Valid Iteration Loop
            print('--------------------------------------')
            print('Validation Start!!')
            print('--------------------------------------')
            model.eval()
            with torch.no_grad():
                for idx, (item, label) in enumerate(tqdm.tqdm(valid_loader, desc=f'NOW EPOCH [{ep} /{self.EPOCH}]')):
                    item = item.to(self.device)
                    label = label.to(self.device)

                    probs = model(item)
                    loss_value = criterion(probs.view(-1).type(torch.FloatTensor), label.view(-1).type(torch.FloatTensor))

                    # loss and acc update
                    probs = probs.cpu().detach().numpy()
                    label = label.cpu().detach().numpy()
                    preds = probs > 0.5
                    batch_acc = (label == preds).mean()

                    valid_loss.append(loss_value.item())
                    valid_acc.append(batch_acc)

                valid_acc_avg = np.mean(valid_acc)
                valid_loss_avg = np.mean(valid_loss)

            print(f'VALID ACC : {valid_acc_avg}')
            print(f'VALID LOSS : {valid_loss_avg}')

            # tensorboard update
            summary.add_scalar("train/acc", train_acc_avg, ep)
            summary.add_scalar("train/loss", train_loss_avg, ep)

            summary.add_scalar("valid/acc", valid_acc_avg, ep)
            summary.add_scalar("valid/loss", valid_loss_avg, ep)

            # save checkpoint
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optim_adamw_state_dict": optim_adamw.state_dict(),
                    "scheduler_cosin_state_dict": scheduler_cosin.state_dict(),
                    "epoch": ep,
                },
                os.path.join(f"{self.OUTPUT_CKP}", f"{ep}.pth"),
            )
