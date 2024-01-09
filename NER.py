from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, BertPreTrainedModel, BertModel
import torch
from torch import Tensor, nn
import numpy as np


categories = set()
CHECKPOINT = "bert-base-chinese"
TOKENIZER = AutoTokenizer.from_pretrained(CHECKPOINT)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

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
    
def label_mapping_dict() -> dict:
    '''get dict for label-to-id'''
    id2label = {0:'O'}
    for c in list(sorted(categories)):
        id2label[len(id2label)] = f"B-{c}"
        id2label[len(id2label)] = f"I-{c}"
    label2id = {v: k for k, v in id2label.items()}
    return label2id

def collate_fn(batch_samples) -> (AutoTokenizer, Tensor):
    '''batch data load function'''
    batch_sentences, batch_tags = [], []
    label2id = label_mapping_dict()
    for sample in batch_samples:
        batch_sentences.append(sample['sentence'])
        batch_tags.append(sample['labels'])
    batch_inputs = TOKENIZER(
        batch_sentences,
        padding=True,
        truncation=True,
        return_tensors="pt" 
    )
    batch_label = np.zeros(batch_inputs['input_ids'].shape, dtype=int)
    for idx, sentence in enumerate(batch_sentences):
        encoding = TOKENIZER(sentence, truncation=True)
        batch_label[idx][0] = -100
        batch_label[idx][len(encoding.tokens())-1:] = -100
        for char_start, char_end, _, tag in batch_tags[idx]:
            token_start = encoding.char_to_token(char_start)
            token_end = encoding.char_to_token(char_end)
            batch_label[idx][token_start] = label2id[f"B-{tag}"]
            batch_label[idx][token_start+1:token_end+1] = label2id[f"I-{tag}"]
    return batch_inputs, torch.tensor(batch_label)

class BertForNER(BertPreTrainedModel):
    """define the model"""
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, len())

if __name__ == '__main__':
    train_data = PeopleDaily('data/china-people-daily-ner-corpus/example.train')
    valid_data = PeopleDaily('data/china-people-daily-ner-corpus/example.dev')
    test_data = PeopleDaily('data/china-people-daily-ner-corpus/example.test')
    # print(test_data[0])
    # print(label_mapping_dict())
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=collate_fn)

    batch_X, batch_y = next(iter(train_dataloader))
    # print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
    # print('batch_y shape:', batch_y.shape)
    # print(batch_X)
    # print(batch_y)
