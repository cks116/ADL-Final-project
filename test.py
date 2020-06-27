import config
import dataset
import function
import os
import torch
import json
import torch.nn as nn
import numpy as np
import pandas as pd

from config import args
from model import BERTBaseJapanese


def predict(path):
    # path = config.VALIDATE_PATH
    # path = config.TESTING_PATH 
    # path = args.testpath
    data = dataset.read_data(path)

    processed_data = dataset.clean_test_data(data)
    valid_dataset = dataset.TagTestDataset(processed_data)

    processed_data = dataset.clean_test_data(data, False)
    predict_dataset = dataset.PredictTestDataset(processed_data)

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = config.VALID_BATCH_SIZE
    )

    predict_dataloader = torch.utils.data.DataLoader(
        predict_dataset,
        batch_size = config.VALID_BATCH_SIZE
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag_model = BERTBaseJapanese()
    tag_model = nn.DataParallel(tag_model)
    tag_model.load_state_dict(torch.load(config.TAG_MODEL_PATH))
    tag_model = tag_model.to(device)

    predict_model = BERTBaseJapanese()
    predict_model = nn.DataParallel(predict_model)
    predict_model.load_state_dict(torch.load(config.PREDICT_MODEL_PATH))
    predict_model = predict_model.to(device)

    function.predict_fn(valid_dataloader, predict_dataloader,tag_model, predict_model, device)


if __name__ == "__main__":
    predict(args.testpath)
