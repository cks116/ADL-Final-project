import utils
import config
import torch
import json
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm

import pickle

def train_fn(data_loader, model, optimizer, device, warmup):
    model.train()
    losses = utils.AverageMeter()
    update = tqdm(data_loader, total = len(data_loader))

    for d in update:
        index = d["text_index"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        tag = d["tag"]
        
        # input
        index = index.to(device, dtype = torch.long)
        token_type_ids = token_type_ids.to(device, dtype = torch.long)
        mask = mask.to(device, dtype = torch.long)
        # target
        targets_start = targets_start.to(device, dtype = torch.long)
        targets_end = targets_end.to(device, dtype = torch.long)
        tag = tag.to(device, dtype = torch.long)


        optimizer.zero_grad()

        output1, output2, ans = model(
            index = index,
            mask = mask,
            token_type_ids = token_type_ids
        )

        loss = loss_fn(output1, output2, ans, targets_start, targets_end, tag, device)
        loss.backward()

        optimizer.step()
        warmup.step()

        losses.update(loss.item(), index.size(0))
        update.set_postfix(loss = losses.avg)

def eval_fn(data_loader, model, device):
    
    model.eval()

    total_fname = []
    total_text = []
    total_index = []
    total_output_start = []
    total_output_end = []
    total_answerable = []

    with torch.no_grad():
        for d in data_loader:
            fname = d["fname"]
            index = d["text_index"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]


            index = index.to(device, dtype = torch.long)
            token_type_ids = token_type_ids.to(device, dtype = torch.long)
            mask = mask.to(device, dtype = torch.long)

            output1, output2, ans = model(
                index = index,
                mask = mask,
                token_type_ids = token_type_ids
            )


            total_output_start.append(output1.cpu())
            total_output_end.append(output2.cpu())
            total_answerable.extend(torch.relu(ans))

            total_index.extend(index)
            total_fname.extend(fname)

    total_output_start = np.vstack(total_output_start)
    total_output_end = np.vstack(total_output_end)

    for num in range(len(total_index)):
        fname = total_fname[num]
        index = total_index[num]
        answerable = total_answerable[num]
        output_start = total_output_start[num]
        output_end = total_output_end[num]

        assert len(index) == len(output_start)
        sep = (index == 3).nonzero()[0]


        output_start = output_start[sep:].argsort()[::-1][0:10]
        output_end = output_end[sep:].argsort()[::-1][0:10]

        output = [0] * len(index[sep:])

        start = output_start[0]
        end = output_end[0]

        for i in output_start:
            for j in output_end:
                if j - i < 60:
                    start = i
                    end = j
                    break
            if end - start < 60:
                break
        
        for i in range(start, end + 1):
            output[i] = 1

        start_tokens = config.TOKENIZER.convert_ids_to_tokens(index[sep:sep + start])
        start_tokens = [x for x in start_tokens if x != "[SEP]" and x != "[CLS]"]
        start_tokens = [x if x != "[UNK]" else "的" for x in start_tokens]


        output_tokens = [x for i, x in enumerate(config.TOKENIZER.convert_ids_to_tokens(index[sep:])) if output[i] == 1]
        output_tokens = [x for x in output_tokens if x != "[SEP]" and x != "[CLS]"]
        output_tokens = [x if x != "[UNK]" else "的" for x in output_tokens]


        start_output = ""
        for token in start_tokens:
            if token.startswith("##"):
                start_output = start_output + token[2:]
            else:
                start_output = start_output + token
        start_output = start_output.strip()

        final_output = ""
        for token in output_tokens:
            if token.startswith("##"):
                final_output = final_output + token[2:]
            else:
                final_output = final_output + token

        final_output = final_output.strip()

        # cross = len(final_output)
        # startt = len(start_output)

        # print(startt)
        # print(text)
        # print(text[startt:startt + cross])
        # print(final_output)
        # input()

        tag = int(answerable.cpu().argmax(0))

        if tag:
            print(answerable, final_output)


def predict_fn(valid_dataloader, predict_dataloader, tag_model, predict_model, device):
    
    tag_model.eval()
    predict_model.eval()

    total_fname = []
    total_text = []
    total_line = []
    total_index = []
    total_output_start = []
    total_output_end = []
    total_answerable = []

    # with open("./predict_test.pkl", "rb") as f:
    #     data = pickle.load(f)

    # for i in range(len(data)):
    #     data[i] = data[i].tolist()

    # data = np.array(data)
    # pk = 0

    with torch.no_grad():
        for d, v in zip(valid_dataloader, predict_dataloader):
            fname = d["fname"]
            index = d["text_index"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]

            index = index.to(device, dtype = torch.long)
            token_type_ids = token_type_ids.to(device, dtype = torch.long)
            mask = mask.to(device, dtype = torch.long)

            output1, output2, ans = tag_model(
                index = index,
                mask = mask,
                token_type_ids = token_type_ids
            )

            # output1, output2, ans2 = tag2_model(
            #     index = index,
            #     mask = mask,
            #     token_type_ids = token_type_ids
            # )

            # output1, output2, ans3 = tag3_model(
            #     index = index,
            #     mask = mask,
            #     token_type_ids = token_type_ids
            # )

            # output1, output2, ans4 = tag4_model(
            #     index = index,
            #     mask = mask,
            #     token_type_ids = token_type_ids
            # )

            # ans_in = torch.div((ans + ans2 + ans3 + ans4), 4)


            # ans = torch.tensor(data[pk*config.VALID_BATCH_SIZE:(pk+1)*config.VALID_BATCH_SIZE])
            # pk += 1


            # new_fname, new_index, new_token_type_ids, new_mask = process_tag(ans, v)
            new_fname, new_text, new_line, new_index, new_token_type_ids, new_mask = process_tag(ans, v)


            if len(new_index) == 0:
                continue

            output1, output2, ans = predict_model(
                index = new_index,
                mask = new_mask,
                token_type_ids = new_token_type_ids
            )

            total_output_start.append(output1.cpu())
            total_output_end.append(output2.cpu())
            total_answerable.extend(torch.relu(ans))

            total_index.extend(new_index)
            total_fname.extend(new_fname)
            total_line.extend(new_line)
            total_text.extend(new_text)

    total_output_start = np.vstack(total_output_start)
    total_output_end = np.vstack(total_output_end)

    pre_line = -1
    pre_fname = -1
    
    ID = []
    Prediction = []
    combine_index = []
    combine_tag = []
    combine_value = []



    for num in range(len(total_index)):
        fname = total_fname[num]
        text = total_text[num]
        line = total_line[num]
        index = total_index[num]
        answerable = total_answerable[num]
        output_start = total_output_start[num]
        output_end = total_output_end[num]

        assert len(index) == len(output_start)
        sep = (index == 3).nonzero()[0]


        output_start = output_start[sep:].argsort()[::-1][0:10]
        output_end = output_end[sep:].argsort()[::-1][0:10]

        output = [0] * len(index[sep:])

        start = output_start[0]
        end = output_end[0]

        for i in output_start:
            for j in output_end:
                if j - i < 60:
                    start = i
                    end = j
                    break
            if end - start < 60:
                break
        
        for i in range(start, end + 1):
            output[i] = 1

        start_tokens = config.TOKENIZER.convert_ids_to_tokens(index[sep:sep + start])

        start_tokens = [x for x in start_tokens if x != "[SEP]" and x != "[CLS]"]

        start_tokens = [x if x != "[UNK]" else "的" for x in start_tokens]


        output_tokens = [x for i, x in enumerate(config.TOKENIZER.convert_ids_to_tokens(index[sep:])) if output[i] == 1]
        output_tokens = [x for x in output_tokens if x != "[SEP]" and x != "[CLS]"]
        output_tokens = [x if x != "[UNK]" else "的" for x in output_tokens]


        start_output = ""
        for token in start_tokens:
            if token.startswith("##"):
                start_output = start_output + token[2:]
            else:
                start_output = start_output + token

        start_output = start_output.strip()

        final_output = ""
        for token in output_tokens:
            if token.startswith("##"):
                final_output = final_output + token[2:]
            else:
                final_output = final_output + token

        final_output = final_output.strip()
        ids = int(answerable.cpu().argmax(0))

        cross = len(final_output)
        final_output = (text[len(start_output): len(start_output) + cross])


        if ids:
            tag = num2tag(ids)
            if pre_line == -1 or pre_line == line:
                pre_line = line
                pre_fname = fname
                combine_index.append(ids)
                combine_tag.append(tag)
                combine_value.append(final_output)
            elif pre_line != line:
                order = np.array(combine_index).argsort()
                final_tag = np.array(combine_tag)[order].tolist()
                final_value = np.array(combine_value)[order].tolist()

                text = ""
                for t, v in zip(final_tag, final_value):
                    if not v: break
                    text += f"{t}:{v} "
                ID.append(f"{pre_fname}-{pre_line}")

                text = text if text else "NONE"
                Prediction.append(text)

                combine_index = [ids]
                combine_tag = [tag]
                combine_value= [final_output]

                pre_line = line
                pre_fname = fname
            
            if num + 1 == len(total_index):
                order = np.array(combine_index).argsort()
                final_tag = np.array(combine_tag)[order].tolist()
                final_value = np.array(combine_value)[order].tolist()

                text = ""
                for t, v in zip(final_tag, final_value):
                    if not v: break
                    text += f"{t}:{v} "

                text = text if text else "NONE"
                ID.append(f"{pre_fname}-{pre_line}")
                Prediction.append(text)

                combine_index = []
                combine_tag = []
                combine_value = []
                pre_line = -1
                pre_fname = -1
        else:
            order = np.array(combine_index).argsort()
            final_tag = np.array(combine_tag)[order].tolist()
            final_value = np.array(combine_value)[order].tolist()

            text = ""
            for t, v in zip(final_tag, final_value):
                if not v: break
                text += f"{t}:{v} "


            if len(order):
                ID.append(f"{pre_fname}-{pre_line}")
            else:
                ID.append(f"{pre_fname}-{pre_line}")
                # text = "NONE"

            text = text if text else "NONE"
            Prediction.append(text)

            combine_index = []
            combine_tag = []
            combine_value = []
            pre_line = line
            pre_fname = fname

            if num + 1 == len(total_index):
                order = np.array(combine_index).argsort()
                final_tag = np.array(combine_tag)[order].tolist()
                final_value = np.array(combine_value)[order].tolist()

                text = ""
                for t, v in zip(final_tag, final_value):
                    if not v: break
                    text += f"{t}:{v} "
                text = "NONE" if not text else ""
                ID.append(f"{pre_fname}-{pre_line}")
                Prediction.append(text)

                combine_index = []
                combine_tag = []
                combine_value = []
                pre_line = -1
                pre_fname = -1



    ID = ID[1:]
    Prediction = Prediction[1:]

    reorder = sorted(range(len(ID)), key=lambda k: ID[k][:9])
    ID = np.array(ID)[reorder].tolist()
    Prediction = np.array(Prediction)[reorder].tolist()

    data = {"ID":ID, "Prediction":Prediction}

    df = pd.DataFrame(data)

    df.to_csv("prediction.csv", index=False)


def loss_fn(o1, o2, ans, t1, t2, answerable, device):
    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)

    # target = torch.zeros((answerable.size(0), 21)).to(device)
    # for i in range(answerable.size(0)):
    #     target[i, answerable[i]] = 1
    # l3 = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(21))(ans, answerable)

    l3 = nn.CrossEntropyLoss()(ans, answerable)

    return l1 + l2 + l3

def num2tag(num):
    table = {"調達年度" : 1, "都道府県" : 2, "入札件名" : 3, "施設名" : 4, "需要場所(住所)" : 5, "調達開始日" : 6,
            "調達終了日" : 7, "公告日": 8, "仕様書交付期限" : 9, "質問票締切日時" : 10, "資格申請締切日時" : 11,
            "入札書締切日時" : 12, "開札日時" : 13, "質問箇所所属/担当者" : 14, "質問箇所TEL/FAX" : 15, "資格申請送付先" : 16,
            "資格申請送付先部署/担当者名" : 17, "入札書送付先" : 18, "入札書送付先部署/担当者名" : 19, "開札場所" : 20}

    return list(table.keys())[list(table.values()).index(num)]

def strB2Q(ustring):
    """把字串半形轉全形"""
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 32:  # 全形空格直接轉換
                inside_code = 12288
            elif (inside_code >= 33 and inside_code <= 126):  # 全形字元（除空格）根據關係轉化
                inside_code += 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)

def process_tag(ans, v):

    fname = v["fname"]
    page_no = v["page_no"]
    text = v["text"]
    line = v["line"]
    index = v["text_index"]
    token_type_ids = v["token_type_ids"]
    mask = v["mask"]
    mask_index = v["mask_index"]


    max_value, max_index = ((nn.Softmax(dim = 1)(ans)).cpu().topk(4, dim = 1))
    # print(max_value)
    # print(max_index)
    # input()

    max_value = max_value.tolist()
    max_index = max_index.tolist()


    new_fname = []
    new_text = []
    new_line = []
    new_index = []
    new_token_type_ids = []
    new_mask = []


    for row, (values, idss) in enumerate(zip(max_value, max_index)):
        biggest = values[0]
        for col, (value, ids) in enumerate(zip(values, idss)):
            # if ids == 0:
            #     break
            if col != 0 and ids == 0:
                break
            temp = index[row]
            # if value >= biggest - 0.1:
            if value >= biggest * 0.4:
                token = config.TOKENIZER.encode(str(ids))[1:-1]
                temp[mask_index[row]] = token[0]

                new_index.append(temp.tolist())
                new_token_type_ids.append(token_type_ids[row].tolist())
                new_mask.append(mask[row].tolist())
                new_fname.append(fname[row])
                new_line.append(line[row])
                new_text.append(text[row])


    new_index = torch.tensor(new_index, dtype = torch.long)
    new_token_type_ids = torch.tensor(new_token_type_ids, dtype = torch.long)
    new_mask = torch.tensor(new_mask, dtype = torch.long)

    return new_fname, new_text, new_line, new_index, new_token_type_ids, new_mask
