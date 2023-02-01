import subprocess
import pandas as pd
import numpy as np
import random
import argparse
import os
from datasets import load_from_disk
from utils import wrap_code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/')
    args = parser.parse_args()
    data_path = f'datasets/{args.data_path}/test'
    dataset = load_from_disk(data_path)
    coc = [len(i) for i in dataset['input_code']]
    coc = np.array(coc) / max(coc)
    rand = [random.random() for i in range(len(coc))]
    dataset = dataset.add_column('coc',coc)
    dataset = dataset.add_column('random',rand)
    print(dataset[0])
    dataset.save_to_disk(f'datasets/{args.data_path}/test_with_metrics')

    


