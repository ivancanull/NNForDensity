import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .transforms import Compose
class OnGPUDataset(Dataset):

    def __init__(self,
                 mode,
                 dataset_dir: str,
                 input_data_format_postfix: str = 'npy',
                 output_data_format_postfix: str = 'npy',
                 train_path=None,
                 val_path=None,
                 test_path=None,
                 transforms=None,
                 separator=' ',
                 device="cpu"):
        
        
        self.dataset_dir = dataset_dir
        self.compose = Compose(transforms=transforms, 
                               mode=mode,
                               input_data_format_postfix=input_data_format_postfix,
                               output_data_format_postfix=output_data_format_postfix)
        self.input_data_format_postfix = input_data_format_postfix
        self.output_data_format_postfix = output_data_format_postfix
        self.file_list = list()
        self.mode = mode
        self.device = device

        if self.mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    self.mode))
        
        self.input_dir = os.path.join(self.dataset_dir, f'input{self.input_data_format_postfix}')
        self.output_dir = os.path.join(self.dataset_dir, f'output{self.output_data_format_postfix}')

        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError('there is not `dataset_dir`: {}.'.format(
                self.dataset_dir))
        
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError('there is not `input_dir`: {}.'.format(
                self.input_dir))
        
        if not os.path.exists(self.output_dir):
            raise FileNotFoundError('there is not `output_dir`: {}.'.format(
                self.output_dir))

        if self.mode == 'train':
            if train_path is None:
                raise ValueError(
                    'When `mode` is "train", `train_path` is necessary, but it is None.'
                )
            elif not os.path.exists(train_path):
                raise FileNotFoundError('`train_path` is not found: {}'.format(
                    train_path))
            else:
                file_path = train_path
        elif self.mode == 'val':
            if val_path is None:
                raise ValueError(
                    'When `mode` is "val", `val_path` is necessary, but it is None.'
                )
            elif not os.path.exists(val_path):
                raise FileNotFoundError('`val_path` is not found: {}'.format(
                    val_path))
            else:
                file_path = val_path
        else:
            if test_path is None:
                raise ValueError(
                    'When `mode` is "test", `test_path` is necessary, but it is None.'
                )
            elif not os.path.exists(test_path):
                raise FileNotFoundError('`test_path` is not found: {}'.format(
                    test_path))
            else:
                file_path = test_path

        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split(separator)
                if len(items) != 2:
                    if self.mode == 'train' or self.mode == 'val':
                        raise ValueError(
                            "File list format incorrect! In training or evaluation task it should be"
                            " image_name{}label_name\\n".format(separator))
                    # 
                    image_path = os.path.join(self.input_dir, items[0])
                    label_path = None
                else:
                    image_path = os.path.join(self.input_dir, items[0])
                    label_path = os.path.join(self.output_dir, items[1])
                self.file_list.append([image_path, label_path])

        data = {}
        # load all the data
        image_npy_concat_path = os.path.join(self.dataset_dir, f"{self.mode}_image_concat.npy")
        label_npy_concat_path = os.path.join(self.dataset_dir, f"{self.mode}_label_concat.npy")
        if os.path.exists(image_npy_concat_path) and os.path.exists(label_npy_concat_path):
            data["img"] = np.load(image_npy_concat_path)
            data["label"] = np.load(label_npy_concat_path)
        else:
            data["img"] = np.hstack([np.load(file_path[0]) for file_path in self.file_list])
            data["label"] = np.hstack([np.load(file_path[1]) for file_path in self.file_list])
            np.save(image_npy_concat_path, data["img"])
            np.save(label_npy_concat_path, data["label"])
        
        # transforms the data
        data = self.compose(data)
        # load all data
        self.input_tensor_concat = torch.from_numpy(data["img"]).to(self.device)
        self.output_tensor_concat = torch.from_numpy(data["label"]).to(self.device)

    def __getitem__(self, idx):
        # todo: return img, label -> tensor (load to device)
        # todo: load dataset
        return self.input_tensor_concat[idx], self.output_tensor_concat[idx]

    def __len__(self):
        return len(self.file_list)