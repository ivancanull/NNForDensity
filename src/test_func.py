from utils import *
from core import *

configs = Config()
configs.load("../configs/test.yaml")

# split dataset
spliter = Spliter(f'../data/{configs.dataset.name}',
                  configs.dataset.dataset_size,
                  configs.dataset.benchmark)

spliter.write_file_path_txt()