from torch.utils.data import Dataset, DataLoader

def data_view(datafile:str, index) -> (int, str):
    '''preview the data file'''
    data = []
    with open(datafile, 'rt', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data.append(line)
    return len(data), data[index]

class LCSTS(Dataset):
    '''
    train_size : first 200000
    valid_size : 1106
    test_size : 10666
    '''
    def __init__(self, src_data_file, tgt_data_file):
        self.data = self.load_data(src_data_file, tgt_data_file)

    def load_data(self, src_data_file:str, tgt_data_file:str) -> dict:
        Data = {}
        src_data, tgt_data = [], []
        with open(src_data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= max_dataset_size:
                    break
                src_data.append(line.rstrip())
        with open(tgt_data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= max_dataset_size:
                    break
                tgt_data.append(line.rstrip())

        for idx, (src, tgt) in enumerate(zip(src_data, tgt_data)):
            Data[idx] = {
                'title':tgt,
                'content':src
            }

        return Data

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    


if __name__ == '__main__':
    '''Dataset'''
    # print(data_view('data/LCSTS/test.tgt.txt', 0))

    max_dataset_size = 200000

    train_dataset = LCSTS('data/LCSTS/train.src.txt', 'data/LCSTS/train.tgt.txt')
    print(next(iter(train_dataset)))