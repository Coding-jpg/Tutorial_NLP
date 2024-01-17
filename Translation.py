from torch.utils.data import Dataset, random_split
import torch
from transformers import AutoModelForSeq2SeqLM
import json

class TransData(Dataset):
    '''
    dataset name: translation2019zh (the first 220000 samples)
    dataset format: {'english': TEXT_A, 'chinese': TEXT_B}
    '''
    def __init__(self, data_file, max_size):
        self.data = self.load_data(data_file, max_size)

    def load_data(self, data_file:str, max_size:int) -> dict:
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= max_size:
                    break
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


    
if __name__ == "__main__":
    """Dataset"""
    max_size = 220000
    train_size = 200000
    valid_size = 20000
    data = TransData("data/translation2019zh/translation2019zh_train.json", max_size)
    train_data, valid_data = random_split(data, [train_size, valid_size]) 
    test_data = TransData("data/translation2019zh/translation2019zh_valid.json", max_size)
    # print(f"train_data size: {len(train_data)}\nvalid_data size: {len(valid_data)}\ntest_data size: {len(test_data)}\nsample:{next(iter(train_data))}")

    """Dataloader"""
    