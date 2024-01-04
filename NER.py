from torch.utils.data import Dataset
from transformers import AutoTokenizer

categories = set()
CHECKPOINT = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

class PeopleDaily(Dataset):
    '''{'sentence':'EXAMPLE', 'labels': [[idx_1,idx_2,'loc_1', 'LOC']]}'''
    def __init__(self, data_file) -> None:
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f.read().split('\n\n')):
                if not line:
                    break
                sentence, labels = '', []
                for i, item in enumerate(line.split('\n')):
                    char, tag = item.split(' ')
                    sentence += char
                    if tag.startswith('B'):
                        labels.append([i, i, char, tag[2:]])
                        categories.add(tag[2:])
                    elif tag.startswith('I'):
                        labels[-1][1] = i
                        labels[-1][2] += char
                Data[idx] = {
                    'sentence': sentence,
                    'labels': labels
                }
        return Data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    train_data = PeopleDaily('data/china-people-daily-ner-corpus/example.train')
    valid_data = PeopleDaily('data/china-people-daily-ner-corpus/example.dev')
    test_data = PeopleDaily('data/china-people-daily-ner-corpus/example.test')
    print(test_data[0])
