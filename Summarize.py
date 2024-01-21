from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AdamW, get_scheduler
import torch
from rouge import Rouge
from tqdm.auto import tqdm
import numpy as np
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

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
    
def collate_fn(batch_samples) -> AutoTokenizer:
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample['content'])
        batch_targets.append(sample['title'])
    batch_data = tokenizer(
        batch_inputs,
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        return_tensors="pt"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch_targets,
            padding=True,
            truncation=True,
            max_length=MAX_TARGET_LENGTH,
            return_tensors="pt"
        )["input_ids"]
        batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
        end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
        for idx, end_idx in enumerate(end_token_index):
            labels[idx][end_idx+1:] = -100
        batch_data['labels'] = labels
    return batch_data

def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1) * len(dataloader)

    model.train()
    for batch, batch_data in enumerate(dataloader, start=1):
        batch_data = batch_data.to(device)
        outputs = model(**batch_data)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss = loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(dataloader, model, mode='Valid'):
    assert mode in ['Valid', 'Test']
    preds, labels = [], []

    model.eval()
    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=MAX_TARGET_LENGTH,
                num_beams=4,
                no_repeat_ngram_size=2,
            ).cpu().numpy()
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        label_tokens = batch_data["labels"]
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        preds += [' '.join(pred.strip()) for pred in decoded_preds]
        labels += [' '.json(label.strip()) for label in decoded_labels]
    scores = rouge.get_scores(hyps=preds, refs=labels, avg=True)
    result = {key: value['f'] * 100 for key, value in scores.items()}
    result['avg'] = np.mean(list(result.values()))
    print(f"{mode} Rouge1: {result['rouge-1']:>0.2f} Rouge2: {result['rouge-2']:>0.2f} RougeL: {result['rouge-l']:>0.2f}\n")
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or Inference")
    parser.add_argument("mode", type=str, help="Train or Inference")
    args = parser.parse_args()
    if args.mode == "train":
        '''Dataset'''
        # print(data_view('data/LCSTS/test.tgt.txt', 0))
        max_dataset_size = 200000

        train_dataset = LCSTS('data/LCSTS/train.src.txt', 'data/LCSTS/train.tgt.txt')
        # print(next(iter(train_dataset)))
        valid_dataset = LCSTS('data/LCSTS/valid.src.txt', 'data/LCSTS/valid.tgt.txt')
        test_dataset = LCSTS('data/LCSTS/test.src.txt', 'data/LCSTS/test.tgt.txt')

        '''DataLoader'''
        MAX_INPUT_LENGTH = 512
        MAX_TARGET_LENGTH = 64
        BATCH_SIZE = 32
        CHECKPOINT = "csebuetnlp/mT5_multilingual_XLSum"
        
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
        model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)
        model = model.to(device)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        # print(next(iter(test_loader)))

        """Train_Valid"""
        rouge = Rouge()
        LR = 2e-5
        EPOCHS = 3
        BEAM_SIZE = 4
        NO_REPEAT_NGRAM_SIZE = 2

        OPTIMIZER = AdamW(model.parameters(), lr=LR)
        LR_SCHEDULER = get_scheduler(
            "linear",
            optimizer=OPTIMIZER,
            num_warmup_steps=0,
            num_training_steps=EPOCHS*len(train_loader)
        )

        total_loss = 0.
        best_avg_rouge = 0.
        for t in range(EPOCHS):
            print(f"Epoch {t+1}/{EPOCHS}\n-------------------------------")
            total_loss = train_loop(train_loader, model, OPTIMIZER, LR_SCHEDULER, EPOCHS, total_loss)
            valid_rouge = test_loop(valid_loader, model, mode='Valid')
            rouge_avg = valid_rouge['avg']
            if rouge_avg > best_avg_rouge:
                best_avg_rouge = rouge_avg
                print('saving new weights...\n')
                torch.save(model.state_dict(), f'epoch_{t+1}_valid_rouge_{rouge_avg:0.4f}_model_weights.bin')
        print("Done!")
    elif args.mode == "infer":
        pass