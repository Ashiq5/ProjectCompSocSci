import os
from typing import Tuple, List
from functools import partial
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, BertPreTrainedModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import OpenAttack
import random
import datasets
from OpenAttack.tags import Tag
from OpenAttack.text_process.tokenizer import PunctTokenizer


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')


class ToxicDataset(Dataset):

    def __init__(self, tokenizer: BertTokenizer, dataframe: pd.DataFrame, lazy: bool = False):
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.pad_token_id
        self.lazy = lazy  # data conversion laziness

        if not self.lazy:
            self.X = []
            self.Y = []
            for i, (row) in tqdm(dataframe.iterrows()):  # convert data into tensor
                x, y = self.row_to_tensor(self.tokenizer, row)
                self.X.append(x)
                self.Y.append(y)
        else:
            self.df = dataframe

    @staticmethod
    def row_to_tensor(tokenizer: BertTokenizer, row: pd.Series) -> Tuple[torch.LongTensor, torch.LongTensor]:
        tokens = tokenizer.encode(row['text'], add_special_tokens=True)
        if len(tokens) > 120:
            tokens = tokens[:119] + [tokens[-1]]
        x = torch.LongTensor(tokens)
        y = torch.LongTensor(row[['label']])
        return x, y

    def __len__(self):
        if self.lazy:
            return len(self.df)
        else:
            return len(self.X)

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        if not self.lazy:
            return self.X[index], self.Y[index]
        else:
            return self.row_to_tensor(self.tokenizer, self.df.iloc[index])


# merges a list of samples to form a mini-batch
def collate_fn(batch: List[Tuple[torch.LongTensor, torch.LongTensor]], device: torch.device) \
        -> Tuple[torch.LongTensor, torch.LongTensor]:
    x, y = list(zip(*batch))
    x = pad_sequence(x, batch_first=True, padding_value=0)  # if batch_fist, B x T x *
    # print('kj', torch.tensor(y).shape)
    y = torch.tensor(y)
    # print('kjk', y.shape)
    return x.to(device), y.to(device)


class BertClassifier(nn.Module):

    def __init__(self, bert: BertModel, num_classes: int):
        super().__init__()
        self.bert = bert
        self.encoder = nn.LSTM(
            input_size=768,  # self.bert.config.hidden_state,
            hidden_size=64,
            bidirectional=True,
        )
        self.classifier = nn.Linear(128, num_classes)  # in_features, out_features

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        self.encoder.flatten_parameters()

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            head_mask=head_mask)
        cls_output = outputs[1]  # batch, hidden

        output, hidden = self.encoder(cls_output)

        cls_output = self.classifier(output)  # batch, 2(classes)
        cls_output = torch.sigmoid(cls_output)  # sigmoid from logit to probability
        # output = torch.max(cls_output, dim=0)[0]
        criterion = nn.CrossEntropyLoss()  # loss function
        loss = 0
        if labels is not None:
            # print(cls_output, labels)
            loss = criterion(cls_output, labels)
        return loss, cls_output


def train(model, iterator, optimizer, scheduler):
    model.train()  # set train mode
    total_loss = 0
    for x, y in tqdm(iterator):
        optimizer.zero_grad()  # set gradients to zero
        mask = (x != 0).float()
        # print("x", x.shape, y.shape)
        loss, outputs = model(x, attention_mask=mask, labels=y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    print(f"Train loss {total_loss / len(iterator)}")


def evaluate(model, iterator):
    model.eval()  # set eval mode
    pred = []
    true = []
    with torch.no_grad():
        total_loss = 0
        correct, total = 0, 0
        for x, y in tqdm(iterator):
            mask = (x != 0).float()
            loss, outputs = model(x, attention_mask=mask, labels=y)
            total_loss += loss
            true += y.cpu().numpy().tolist()
            pred += outputs.cpu().numpy().tolist()

            pred_labels = outputs.argmax(dim=-1)
            correct += (pred_labels == y).sum().item()
            total += len(pred_labels)
    print("Evaluate accuacy", correct / total)
    true = np.array(true)
    pred = np.array(pred)
    # for i, name in enumerate(['label']):
    #   print(f"{name} roc_auc {roc_auc_score(true[i], pred[:, i])}")
    print(f"Evaluate loss {total_loss / len(iterator)}")


def load_model_from_disk():
    model.load_state_dict(torch.load(
        "/content/drive/MyDrive/ColabNotebooks/models/lstm-bert-kaggle.pt"))  # Todo: provide file name here


# configure access interface of the customized victim model by extending OpenAttack.Classifier.
class MyClassifier(OpenAttack.Classifier):
    def __init__(self, model):
        self.model = model

    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)

    # access to the classification probability scores with respect input sentences
    def get_prob(self, input_):
        print(input_, self.model(input_))
        loss, outputs = self.model(input_)
        return outputs


def attack(model_to_attack, dataset_on_attack, attacker_class):
    attack_eval = OpenAttack.AttackEval(attacker_class, model_to_attack, metrics=[
        OpenAttack.metric.SemanticSimilarity(),
        OpenAttack.metric.ModificationRate()
    ])
    # attack_eval.eval(dataset, visualize=True)

    correct_samples = [
        inst for inst in dataset_on_attack if model_to_attack.get_pred([inst["x"]])[0] == inst["y"]
    ]

    accuracy = len(correct_samples) / len(dataset_on_attack)

    adversarial_samples = {
        "x": [],
        "y": [],
        "tokens": []
    }

    for result in tqdm.tqdm(attack_eval.ieval(correct_samples), total=len(correct_samples)):
        if result["success"]:
            adversarial_samples["x"].append(result["result"])
            adversarial_samples["y"].append(result["data"]["y"])
            adversarial_samples["tokens"].append(tokenizer.tokenize(result["result"], pos_tagging=False))

    attack_success_rate = len(adversarial_samples["x"]) / len(correct_samples)

    print("Accuracy: %lf%%\nAttack success rate: %lf%%" % (accuracy * 100, attack_success_rate * 100))

    return datasets.Dataset.from_dict(adversarial_samples)



if __name__ == "__main__":
    # load training and test data
    input_file_path = '/content/drive/MyDrive/ColabNotebooks/dataset/'  # './input/'
    train_df = pd.read_csv(os.path.join(input_file_path, 'kaggle-train.csv'))
    train_split_df, val_split_df = train_test_split(train_df, test_size=0.05)
    test_df1 = pd.read_csv(os.path.join(input_file_path, 'kaggle-test.csv'))[0:7000]
    test_df2 = pd.read_csv(os.path.join(input_file_path, 'kaggle-test.csv'))[7000:]
    print(train_df.shape)
    print(list(train_df.columns))
    print(train_split_df.shape)
    print(val_split_df.shape)
    print(test_df1.shape)
    print(test_df2.shape)

    # load tokenizer
    bert_model_name = 'bert-base-uncased'  # @param {type:"string"}
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    assert tokenizer.pad_token_id == 0, "Padding vlaue used in masks is set to zero, please change it everywhere"

    # load dataset
    train_dataset = ToxicDataset(tokenizer, train_split_df, lazy=True)
    dev_dataset = ToxicDataset(tokenizer, val_split_df, lazy=True)
    test_dataset = ToxicDataset(tokenizer, test_df1, lazy=True)
    test_dataset_for_attack = ToxicDataset(tokenizer, test_df2, lazy=True)

    collate_fn = partial(collate_fn, device=device)

    BATCH_SIZE = 32
    train_sampler = RandomSampler(train_dataset)
    dev_sampler = RandomSampler(dev_dataset)
    test_sampler = RandomSampler(test_dataset)
    test_sampler_for_attack = RandomSampler(test_dataset_for_attack)

    # Dataset, Sampler, collate_fn -> DataLoader
    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                sampler=train_sampler, collate_fn=collate_fn)
    dev_iterator = DataLoader(dev_dataset, batch_size=BATCH_SIZE,
                              sampler=dev_sampler, collate_fn=collate_fn)
    test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                               sampler=test_sampler, collate_fn=collate_fn)
    test_iterator_for_attack = DataLoader(test_dataset_for_attack, batch_size=BATCH_SIZE,
                                          sampler=test_sampler_for_attack, collate_fn=collate_fn)
    print(len(train_iterator), len(dev_iterator), len(test_iterator), len(test_iterator_for_attack),
          len(test_iterator_for_attack))

    # load model
    model = BertClassifier(BertModel.from_pretrained(bert_model_name), 2).to(device)

    # load others
    no_decay = ['bias', 'LayerNorm.weight']  # no deacy parameters
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    EPOCH_NUM = 10

    # https://paperswithcode.com/method/slanted-triangular-learning-rates
    # triangular learning rate; linearly grows until half of first epoch, then linearly decays
    warmup_steps = 10 ** 3
    total_steps = len(train_iterator) * EPOCH_NUM - warmup_steps
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
    # https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # train or load model from disk
    training_done = True
    if training_done:
        load_model_from_disk()
    else:
        for i in range(EPOCH_NUM):
            print('=' * 50, f"EPOCH {i}", '=' * 50)
            train(model, train_iterator, optimizer, scheduler)
            evaluate(model, dev_iterator)

            output_dir = "/content/drive/MyDrive/ColabNotebooks/models/"
            model_name = "lstm-bert-kaggle"
            model_path = output_dir + model_name + ".pt"
            torch.save(model.state_dict(), model_path)

    # evaluate
    evaluate(model, test_iterator)

    # choose attack framework
    attack_framework = 'openattack'

    # apply attack, measure performance, and save generated adv. samples for adv. training
    OpenAttack.DataManager.enable_cdn()
    victim = MyClassifier(model)
    test_text, test_labels = test_df2['text'], test_df2['label']
    dataset_for_attack = datasets.Dataset.from_dict({
        "x": list(test_text),
        "y": list(test_labels)
    })

    attacker_tf = OpenAttack.attackers.TextFoolerAttacker()
    attacker_tb = OpenAttack.attackers.TextBuggerAttacker()
    attacker_bae = OpenAttack.attackers.BAEAttacker()
    x = attack(victim, dataset_for_attack, attacker_tf)
    y = attack(victim, dataset_for_attack, attacker_tb)
    z = attack(victim, dataset_for_attack, attacker_bae)
    adversarial_samples = x  # TODO: change here (could be y+z for transferable AT)
    # note down performance

    # adversarial training
    # augment training dataset with adv. samples
    new_dataset = {
        "x": [],
        "y": [],
        "tokens": []
    }
    adv_samples_dataset = ToxicDataset(tokenizer, adversarial_samples, lazy=True)
    adv_sampler = RandomSampler(adv_samples_dataset)
    test_iterator = DataLoader(adv_samples_dataset, batch_size=BATCH_SIZE,
                               sampler=adv_sampler, collate_fn=collate_fn)

    # retrain
    for i in range(EPOCH_NUM):
        print('=' * 50, f"EPOCH {i}", '=' * 50)
        train(model, train_iterator, optimizer, scheduler)
        evaluate(model, dev_iterator)

        output_dir = "/content/drive/MyDrive/ColabNotebooks/models/"
        model_name = "lstm-bert-kaggle-adv-trained"
        model_path = output_dir + model_name + ".pt"
        torch.save(model.state_dict(), model_path)

    finetuned_model = model

    # recheck the robustness by applying attack again and measure robustness
    attack(finetuned_model, dataset_for_attack, attacker_tf)
    attack(finetuned_model, dataset_for_attack, attacker_tb)
    attack(finetuned_model, dataset_for_attack, attacker_bae)



