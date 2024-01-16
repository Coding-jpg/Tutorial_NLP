from torch.utils.data import Dataset

class TransData(Dataset):
    '''dataset format: {'english': TEXT_A, 'chinese': TEXT_B}'''
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file) -> dict:
        Data = {}
        return Data