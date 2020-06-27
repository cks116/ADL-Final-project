import os
import utils
import config
import dataset
import function
import torch
import pandas as pd
import numpy as np
import torch.nn as nn

from param import args
from model import BERTBaseJapanese
from transformers import AdamW, get_linear_schedule_with_warmup


torch.manual_seed(0)

def train():
    train_path = config.TRAINING_PATH
    train_data = dataset.read_data(train_path)

    val_path = config.VALIDATE_PATH
    val_data = dataset.read_data(val_path)


    if args.largeset:
        train_data = train_data + val_data


    if args.tag and args.predict:
        print(f'Get wrong input\n')
        exit()
    
    if args.tag:
        processed_train_data = dataset.clean_data(train_data)
        train_dataset = dataset.TagDataset(processed_train_data)

    if args.predict:
        processed_train_data = dataset.clean_data(train_data, False)
        train_dataset = dataset.PredictDataset(processed_train_data)


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config.TRAIN_BATCH_SIZE,
        shuffle = True,
    )

    # processed_val_data = dataset.clean_data(val_data)

    # val_dataset = dataset.PredictDataset(processed_val_data[:100])
    # val_dataset = dataset.TagDataset(processed_val_data[:100])

    # valid_dataloader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size = config.TRAIN_BATCH_SIZE,
    # )

    num_training_steps = len(processed_train_data) / config.TRAIN_BATCH_SIZE * config.EPOCHS


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTBaseJapanese()
    model = nn.DataParallel(model)
    model = model.to(device)


    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
    optimizer = AdamW(optimizer_grouped_parameters, lr= 3e-5)

    warmup = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = num_training_steps
    )


    best = 0
    for epoch in range(config.EPOCHS):
        function.train_fn(train_dataloader, model, optimizer, device, warmup)
        # function.eval_fn(valid_dataloader, model, device)

    if args.tag:
        torch.save(model.state_dict(), config.TAG_MODEL_PATH)

    if args.predict:
        torch.save(model.state_dict(), config.PREDICT_MODEL_PATH)

if __name__ == "__main__":
    train()
