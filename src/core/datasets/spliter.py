import os
import random

from numpy import indices

__all__ = [
    "Spliter",
]

class Spliter():
    def __init__(self, 
                 dataset_dir: str,
                 dataset_size: int,
                 data_prefix: str,
                 input_data_format_postfix: str = 'npy',
                 output_data_format_postfix: str = 'npy',
                 train_ratio: float = 0.8, 
                 test_ratio: float = 0.1,
                 val_ratio: float = None, ):
        
        self.dataset_dir = dataset_dir
        self.dataset_size = dataset_size
        self.data_prefix = data_prefix
        self.input_data_format_postfix = input_data_format_postfix
        self.output_data_format_postfix = output_data_format_postfix
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

        # infer validation ratio
        if val_ratio is None:
            self.val_ratio = 1.0 - self.train_ratio - self.test_ratio

        # the sum of ratio need to one
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 0.0001:
            raise ValueError(f'the sum of train ratio: {self.train_ratio}, val ratio: {self.val_ratio}, and test ratio: {self.test_ratio} must be 1')

        # calculate size of each dataset
        self.train_size = int(self.train_ratio * self.dataset_size)
        self.test_size = int(self.test_ratio * self.dataset_size)
        self.val_size = self.dataset_size - self.train_size - self.test_size
        if self.val_size < 0:
            raise ValueError('val size is negative, decrease train and test ratio')

    def split(self):
        self.input_dir = os.path.join(self.dataset_dir, f'input{self.input_data_format_postfix}')
        self.output_dir = os.path.join(self.dataset_dir, f'output{self.output_data_format_postfix}')

        l = list(range(1, self.dataset_size + 1))
        random.shuffle(l)
        
        self.indices = {}
        self.indices['train'] = l[: self.train_size]
        self.indices['test'] = l[self.train_size : self.train_size + self.test_size]
        self.indices['val'] = l[self.train_size + self.test_size :]
        return self.indices
    
    def write_file_path_txt(self, 
                            regenerate=False):
        self.split()
        # recursively generate dataset indices
        for mode in ['train', 'test', 'val']:
            if os.path.exists(os.path.join(self.dataset_dir, f'{mode}_indices.txt')) and (not regenerate):
                continue
            with open(os.path.join(self.dataset_dir, f'{mode}_indices.txt'), 'w') as f:
                for idx in self.indices[mode]:
                    input_file_dir = f'{self.data_prefix}_input_{idx}.{self.input_data_format_postfix}'
                    output_file_dir = f'{self.data_prefix}_output_{idx}.{self.output_data_format_postfix}'
                    if not os.path.exists(os.path.join(self.input_dir, input_file_dir)):
                        raise FileNotFoundError(f'file {input_file_dir} does not exists')
                    elif not os.path.exists(os.path.join(self.output_dir, output_file_dir)):
                        raise FileNotFoundError(f'file {output_file_dir} does not exists')
                    else:
                        f.write(f"{input_file_dir} {output_file_dir}\n")


            
