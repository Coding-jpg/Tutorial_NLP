from torch.utils.data import Dataset, random_split, DataLoader
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AdamW, get_scheduler
from sacrebleu.metrics import BLEU
import numpy as np

import json
from tqdm.auto import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device.')

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

def collate_fn(batch_samples) -> AutoTokenizer:
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample['chinese'])
        batch_targets.append(sample['english'])
    batch_data = tokenizer(
        batch_inputs,
        padding=True,
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch_targets,
            padding=True,
            truncation=True,
            max_length=max_input_length,
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
        outputs=  model(**batch_data)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(dataloader, model):
    preds, labels = [], []

    model.eval()
    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=max_target_length,
            ).cpu().numpy()
        label_tokens = batch_data["labels"].cpu().numpy()
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        preds += [pred.strip() for pred in decoded_preds]
        labels += [[label.strip()] for label in decoded_labels]
    bleu_score = bleu.corpus_score(preds, labels).score
    print(f"BLEU: {bleu_score:>0.2f}\n")
    return bleu_score

def translate(checkpoint:str, sentence:str):
    model.load_state_dict(torch.load(checkpoint)).to(device)

    model.eval()
    with torch.no_grad():
        input_token_id = tokenizer(
            sentence,
            return_tensors="pt"
        ).to(device)
        # print(f"input_token_id: {input_token_id}")
        generated_token = model.generate(
            input_token_id["input_ids"],
            attention_mask=input_token_id["attention_mask"],
            max_length=128,
        )
        # print(f"generated_token_id:{generated_token}")
        sentence_pred = tokenizer.decode(generated_token[0], skip_special_tokens=True)
        print(sentence_pred)

if __name__ == "__main__":
    # """Dataset"""
    # max_size = 220000
    # train_size = 200000
    # valid_size = 20000

    # data = TransData("data/translation2019zh/translation2019zh_train.json", max_size)
    # train_data, valid_data = random_split(data, [train_size, valid_size]) 
    # test_data = TransData("data/translation2019zh/translation2019zh_valid.json", max_size)
    # # print(f"train_data size: {len(train_data)}\nvalid_data size: {len(valid_data)}\ntest_data size: {len(test_data)}\nsample:{next(iter(train_data))}")

    # """Dataloader"""
    # batch_size = 4
    # max_input_length = 128
    # max_target_length = 128
    # num_workers = 4
    model_checkpoint = "Helsinki-NLP/opus-mt-zh-en"

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    # model = model.to(device)
    # print(f"tokenizer: {tokenizer}\nmodel: {model}")

    # train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    # valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=False, collate_fn=collate_fn)
    # test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=collate_fn)
    # # print(next(iter(valid_dataloader)))

    # """Train_valid loop"""
    # learning_rate = 1e-5
    # epoch_num = 1
    # bleu = BLEU()

    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    # lr_scheduler = get_scheduler(
    #     "linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=epoch_num*len(train_dataloader),
    # )

    # total_loss = 0.
    # best_bleu = 0.
    # for t in range(epoch_num):
    #     print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    #     total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t+1, total_loss)
    #     valid_bleu = test_loop(valid_dataloader, model)
    #     if valid_bleu > best_bleu:
    #         best_bleu = valid_bleu
    #         print('saving new weights...\n')
    #         torch.save(
    #             model.state_dict(),
    #             f'epoch_{t+1}_valid_bleu_{valid_bleu:0.2f}_model_weights.bin'
    #         )
    # print("Done!")

    """Test"""
    checkpoint_ft = 'epoch_1_valid_bleu_27.60_model_weights.bin'
    chinese_sentence = '这是一个翻译测试'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    translate(checkpoint_ft, chinese_sentence)