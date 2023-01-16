import os
import os.path as osp
import scipy.io as scio
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MagDataset(Dataset):

    def __init__(self, data_dir, ann_file, is_train=True, aug={}):

        self.dicts = []
        self.aug = aug

        files = os.listdir(data_dir)
        with open(ann_file, 'r', encoding='utf-8') as f:
            labels = f.read().split('\n')
        print(f'read data guide by {ann_file}')
        for label in tqdm(labels):
            x_path = f'{data_dir}/magnetic_anomaly_{label}.mat'
            y_path = f'{data_dir}/magnetic_{label}.mat'
            x_img = scio.loadmat(x_path)['ma']/400
            y_img = scio.loadmat(y_path)['mag']*25
            x_img = x_img.astype(np.float32)
            y_img = y_img.astype(np.float32).transpose((2, 0, 1))
            x_tensor = torch.from_numpy(x_img).unsqueeze(0)
            y_tensor = torch.from_numpy(y_img).unsqueeze(0)
            self.dicts.append(
                dict(mag2d=x_tensor, mag3d=y_tensor, label=label))

    def __getitem__(self, index: int):
        data = self.dicts[index]
        flip_dims = []
        for k, v in self.aug.items():
            if k == 'horizontal':
                if np.random.random() < v:
                    flip_dims.append(-1)
            elif k == 'vertical':
                if np.random.random() < v:
                    flip_dims.append(-2)
            elif k == 'transpose':
                if np.random.random() < v:
                    data['mag3d'] = data['mag3d'].transpose(-2, -1)
                    data['mag2d'] = data['mag2d'].transpose(-2, -1)
            else:
                raise ValueError(f'augmentation not support {k}')
            if len(flip_dims) > 0:
                data['mag3d'] = data['mag3d'].flip(dims=flip_dims)
                data['mag2d'] = data['mag2d'].flip(dims=flip_dims)
        return data

    def __len__(self) -> int:
        return len(self.dicts)


# if __name__ == '__main__':
#     dataset = GraDataset('data/gravity_data',
#                          'data/gravity_ann/train.txt', is_train=True)
#     print(len(dataset))
#     data = dataset.__getitem__(3)
#     print(data.keys())
#     gt_tensor = data['gra']
#     lq_tensor = data['density']
#     print(gt_tensor.shape, gt_tensor.min(), gt_tensor.max())
#     print(lq_tensor.shape, lq_tensor.min(), lq_tensor.max())
