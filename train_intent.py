import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from tqdm import trange

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_loader = DataLoader(dataset=datasets[TRAIN], 
                              batch_size=args.batch_size, 
                              collate_fn=datasets[TRAIN].collate_fn)
    val_loader = DataLoader(dataset=datasets[DEV], 
                              batch_size=args.batch_size, 
                              collate_fn=datasets[DEV].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    device = args.device
    model = SeqClassifier(embeddings=embeddings,
                          hidden_size=args.hidden_size,
                          num_layers=args.num_layers,
                          dropout=args.dropout,
                          bidirectional=args.bidirectional,
                          num_class=len(intent2idx) # 150
    )
                          
    model = model.to(device)
    # loss function
    criterion = nn.CrossEntropyLoss() 
    # TODO: init optimizer
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), 
                            lr=args.lr, 
                            weight_decay=args.weight_decay,
    )

    best_acc = 0.0
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for num_epoch in epoch_pbar:
        train_acc, train_loss, val_acc, val_loss = 0.0, 0.0, 0.0, 0.0
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        max_norm = 0 # maximum gradient norm of batches
        for i, data in enumerate(train_loader):
            inputs, labels = data['text'], data['intent']
            labels = [intent2idx[label] for label in labels]

            inputs, labels = torch.LongTensor(inputs), torch.LongTensor(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            batch_loss = criterion(outputs, labels)
            _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
            batch_loss.backward() 
            
            total_norm = 0
            for param in model.parameters():
                param_norm = param.grad.norm(2)
                total_norm += param_norm ** 2
            max_norm = max(max_norm, total_norm)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10) # clipping
            optimizer.step() 

            train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
            train_loss += batch_loss.item()
            
        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data['text'], data['intent']
                labels = [intent2idx[label] for label in labels]

                inputs, labels = torch.LongTensor(inputs), torch.LongTensor(labels)
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                batch_loss = criterion(outputs, labels) 
                _, val_pred = torch.max(outputs, 1) 
            
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                val_loss += batch_loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                args.num_epoch, num_epoch, train_acc/len(datasets[TRAIN]), train_loss/len(train_loader), 
                val_acc/len(datasets[DEV]), val_loss/len(val_loader)
            ))

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), args.ckpt_dir / args.ckpt_name)
                print('saving model with acc {:.3f}'.format(best_acc/len(datasets[DEV])))


    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )
    parser.add_argument("--ckpt_name", type=Path, default="/best.pt")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512) # 512
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1) # 0.1
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128) # 128

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default=100) # 100

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(args)
    main(args)
