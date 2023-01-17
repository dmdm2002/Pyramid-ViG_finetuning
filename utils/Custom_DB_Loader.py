import torch.utils.data as data
import PIL.Image as Image

import pandas as pd
import re


class Loader(data.DataLoader):
    def __init__(self, root, run_type='train', transform=None, aug=False):
        super(Loader, self).__init__(self)
        self.root = root
        self.run_type = run_type
        self.aug = aug

        self.path = f'{self.root}/{self.run_type}'
        self.transform = transform

        if self.aug:
            df = pd.read_csv(f'{self.root}/DB/DB_informations/aug_{self.run_type}.csv')
        else:
            df = pd.read_csv(f'{self.root}/DB/DB_informations/{self.run_type}.csv')
        # print(df)

        if self.run_type == 'test':
            self.path_list = self.get_paths(df)

        else:
            self.label_list = self.get_labels(df)
            self.path_list = self.get_paths(df)

    def get_labels(self, df):
        return df.iloc[:,2:].values

    def get_paths(self, df):
        path_list = df.iloc[:, 0:1].values
        result_path = []

        for path in path_list:
              if self.run_type == 'test':
                folder_name = 'test'

              else:
                # train & valid data는 전부 train folder에 있으니까
                if self.aug:
                    folder_name = 'aug_train'
                else:
                    folder_name = 'train'

              path = f'{self.root}/DB/{folder_name}/{path[0]}.jpg'
              result_path.append(path)

        return result_path

    def __getitem__(self, index):
        if self.run_type == 'test':
            item = self.transform(Image.open(self.path_list[index]))

            return [item, self.path_list[index]]

        else:
            item = self.transform(Image.open(self.path_list[index]))
            label = self.label_list[index]

            return [item, label]

    def __len__(self):
        return len(self.path_list)