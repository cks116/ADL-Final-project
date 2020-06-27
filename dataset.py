import os
import re
import config
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class PredictDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        # page_no = self.data[index].get("page_no")
        fname = self.data[index].get("fname")
        text = self.data[index].get("text")
        orig_text = self.data[index].get("orig_text")
        line = self.data[index].get("index")
        # parent_index = self.data[index].get("parent_index")
        parent_text = self.data[index].get("parent_text")
        # is_title = self.data[index].get("is_title")
        # is_table = self.data[index].get("is_table")
        tag = self.data[index].get("tag")
        # zero = torch.zeros(21)
        # zero[tag] = 1
        # tag = zero

        value = self.data[index].get("value")
        ans_start = self.data[index].get("start")

        if ans_start != -1:
            len_answer = len(value)
        else:
            len_answer = 0

        ans_start += len(parent_text)

        tag_ids = self.tokenizer.encode([str(tag)])
        tok_input_ids = self.tokenizer.encode(parent_text, text)
        tok_input_ids += tag_ids[1:]
        tok_input_tokens = self.tokenizer.convert_ids_to_tokens(tok_input_ids)

        # print(tok_input_tokens)

        # process offset
        tok_input_offsets = []
        last = 0
        for i, token in enumerate(tok_input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                tok_input_offsets.append((0, 0))
            elif token.startswith("##"):
                tok_input_offsets.append((last, last + len(token) - 2))
                last += len(token) - 2
            else:
                tok_input_offsets.append((last, last + len(token)))
                last += len(token)
        tok_input_offsets = tok_input_offsets[1:-1] # remove ["CLS"] ["SEP"]


        # cal start, end
        targets = [0] * (len(tok_input_tokens) - 2)  # orig not have cls sep

        for i, (offset1, offset2) in enumerate(tok_input_offsets):
            if offset1 >= ans_start and offset2 <= ans_start + len_answer:
                targets[i] = 1

        targets = [0] + targets + [0]  # add ["CLS"] ["SEP"]
        non_zero = np.nonzero(targets)[0]

        if len(non_zero) > 0:
            start = non_zero[0]
            end = non_zero[-1]
        else:
            start = end = 0


        mask = [1] * len(tok_input_ids)

        first = tok_input_ids.index(3) + 1
        second = tok_input_ids.index(3, first) - first + 1
        third = len(tok_input_ids) - first - second

        token_type_ids = [0] * first + [0] * second + [1] * third

        padding_len = self.max_len - len(tok_input_ids)

        text_index = tok_input_ids + [0] * padding_len
        mask = mask + [0] * padding_len
        token_type_ids = token_type_ids + [0] * padding_len

        return {
            "fname": fname,
            "text": orig_text,
            "line": line,
            "text_index": torch.tensor(text_index, dtype = torch.long),
            "mask": torch.tensor(mask, dtype = torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype = torch.long),
            "targets_start": start,
            "targets_end": end,
            # "padding_len": torch.tensor(padding_len, dtype = torch.long),
            # "context_question_token": " ".join(tok_input_tokens),
            "tag": tag,
            "mask_index": first + second
        }


class TagDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        page_no = self.data[index].get("page_no")
        fname = self.data[index].get("fname")
        text = self.data[index].get("text")
        # text = str(page_no) + "。" + text
        text = text + "。" + str(page_no)
        # index = self.data[index].get("index")
        # parent_index = self.data[index].get("parent_index")
        parent_text = self.data[index].get("parent_text")

        # if self.data[index - 1]:
        #     parent_text = parent_text + "。" + self.data[index - 1].get("text")

        # print(parent_text)

        # is_title = self.data[index].get("is_title")
        # is_table = self.data[index].get("is_table")
        tag = self.data[index].get("tag")
        # zero = torch.zeros(21)
        # zero[tag] = 1
        # tag = zero
        
        value = self.data[index].get("value")
        ans_start = self.data[index].get("start")

        if ans_start != -1:
            len_answer = len(value)
        else:
            len_answer = 0

        ans_start += len(parent_text)

        tok_input_ids = self.tokenizer.encode(parent_text, text)
        tok_input_tokens = self.tokenizer.convert_ids_to_tokens(tok_input_ids)

        # process offset
        tok_input_offsets = []
        last = 0
        for i, token in enumerate(tok_input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                tok_input_offsets.append((0, 0))
            elif token.startswith("##"):
                tok_input_offsets.append((last, last + len(token) - 2))
                last += len(token) - 2
            else:
                tok_input_offsets.append((last, last + len(token)))
                last += len(token)
        tok_input_offsets = tok_input_offsets[1:-1] # remove ["CLS"] ["SEP"]


        # cal start, end
        targets = [0] * (len(tok_input_tokens) - 2)  # orig not have cls sep

        for i, (offset1, offset2) in enumerate(tok_input_offsets):
            if offset1 >= ans_start and offset2 <= ans_start + len_answer:
                targets[i] = 1

        targets = [0] + targets + [0]  # add ["CLS"] ["SEP"]
        non_zero = np.nonzero(targets)[0]

        if len(non_zero) > 0:
            start = non_zero[0]
            end = non_zero[-1]
        else:
            start = end = 0


        mask = [1] * len(tok_input_ids)

        first = tok_input_ids.index(3) + 1
        second = tok_input_ids.index(3, first) - first + 1
        # third = len(tok_input_ids) - first - second

        token_type_ids = [0] * first + [1] * second

        padding_len = self.max_len - len(tok_input_ids)

        text_index = tok_input_ids + [0] * padding_len
        mask = mask + [0] * padding_len
        token_type_ids = token_type_ids + [0] * padding_len

        return {
            "fname": fname,
            "text_index": torch.tensor(text_index, dtype = torch.long),
            "mask": torch.tensor(mask, dtype = torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype = torch.long),
            "targets_start": start,
            "targets_end": end,
            # "padding_len": torch.tensor(padding_len, dtype = torch.long),
            # "context_question_token": " ".join(tok_input_tokens),
            "tag": tag
        }

class PredictTestDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        

        fname = self.data[index].get("fname")
        page_no = self.data[index].get("page_no")
        text = self.data[index].get("text")
        orig_text =self.data[index].get("orig_text")
        line = self.data[index].get("index")
        parent_text = self.data[index].get("parent_text")
        tag = ["MASK"]


        tag_ids = self.tokenizer.encode(tag)
        tok_input_ids = self.tokenizer.encode(parent_text, text)
        tok_input_ids += tag_ids[1:]
        tok_input_tokens = self.tokenizer.convert_ids_to_tokens(tok_input_ids)


        mask = [1] * len(tok_input_ids)

        first = tok_input_ids.index(3) + 1
        second = tok_input_ids.index(3, first) - first + 1
        third = len(tok_input_ids) - first - second

        token_type_ids = [0] * first + [0] * second + [1] * third

        padding_len = self.max_len - len(tok_input_ids)

        text_index = tok_input_ids + [0] * padding_len
        mask = mask + [0] * padding_len
        token_type_ids = token_type_ids + [0] * padding_len

        return {
            "fname": fname,
            "page_no": page_no,
            "text": orig_text,
            "line": line,
            "text_index": torch.tensor(text_index, dtype = torch.long),
            "mask": torch.tensor(mask, dtype = torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype = torch.long),
            # "padding_len": torch.tensor(padding_len, dtype = torch.long),
            # "context_question_token": " ".join(tok_input_tokens),
            "mask_index": first + second
        }

class TagTestDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        page_no = self.data[index].get("page_no")
        fname = self.data[index].get("fname")
        text = self.data[index].get("text")
        # text = str(page_no) + "。" + text
        text = text + "。" + str(page_no)
        parent_text = self.data[index].get("parent_text")

        # if self.data[index - 1]:
        #     parent_text = parent_text + "。" + self.data[index - 1].get("text")

        # print(parent_text)        

        tok_input_ids = self.tokenizer.encode(parent_text, text)
        tok_input_tokens = self.tokenizer.convert_ids_to_tokens(tok_input_ids)


        mask = [1] * len(tok_input_ids)

        first = tok_input_ids.index(3) + 1
        second = tok_input_ids.index(3, first) - first + 1
        # third = len(tok_input_ids) - first - second

        token_type_ids = [0] * first + [1] * second

        padding_len = self.max_len - len(tok_input_ids)

        text_index = tok_input_ids + [0] * padding_len
        mask = mask + [0] * padding_len
        token_type_ids = token_type_ids + [0] * padding_len

        return {
            "fname": fname,
            "text_index": torch.tensor(text_index, dtype = torch.long),
            "mask": torch.tensor(mask, dtype = torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype = torch.long),
        }



def clean_data(data, Tag = True):
    clean = []

    length = len(data)

    table = {"調達年度" : 1, "都道府県" : 2, "入札件名" : 3, "施設名" : 4, "需要場所（住所）" : 5, "調達開始日" : 6,
            "調達終了日" : 7, "公告日": 8, "仕様書交付期限" : 9, "質問票締切日時" : 10, "資格申請締切日時" : 11,
            "入札書締切日時" : 12, "開札日時" : 13, "質問箇所所属／担当者" : 14, "質問箇所ＴＥＬ／ＦＡＸ" : 15, "資格申請送付先" : 16,
            "資格申請送付先部署／担当者名" : 17, "入札書送付先" : 18, "入札書送付先部署／担当者名" : 19, "開札場所" : 20}


    for i, items in enumerate(data):
    # for items in data:
        inputs = {}
        parent_text = ""

        if items[4] == 'x' : items[4] = 1
        if items[5] == 'x' : items[5] = 1

        if items[3]:
            parent = data[int(items[3] - 1)]
            parent_text = "。" + parent[1].replace(" ", "")
            parent_text = re.sub('＊|\*|\s+|①|②|◎|③|※|④', '', parent_text)
            if parent[3]:
                grandparent = data[int(parent[3]) - 1]
                parent_text = grandparent[1].replace(" ", "") + parent_text
                parent_text = re.sub('＊|\*|\s+|①|②|◎|③|※|④', '', parent_text)
                parent_text = strQ2B(parent_text)

            if Tag:
                if i - 2 >= 0:
                    previous = data[i - 2]
                    previous_text = strQ2B(previous[1].replace(" ", ""))
                    previous_text = re.sub('＊|\*|\s+|①|②|◎|③|※|④', '', previous_text)
                    parent_text = parent_text + "。" + previous_text
                else:
                    parent_text = parent_text + "。"

                if i - 1 >= 0:
                    previous = data[i - 1]
                    previous_text = strQ2B(previous[1].replace(" ", ""))
                    previous_text = re.sub('＊|\*|\s+|①|②|◎|③|※|④', '', previous_text)
                    parent_text = parent_text + "。" + previous_text
                else:
                    parent_text = parent_text + "。"

                    
                if i + 1 < length:
                    foward = data[i + 1]
                    foward_text = strQ2B(foward[1].replace(" ", ""))
                    foward_text = re.sub('＊|\*|\s+|①|②|◎|③|※|④', '', foward_text)
                    parent_text = parent_text + "。" + foward_text
                else:
                    parent_text = parent_text + "。"

                if i + 2 < length:
                    foward = data[i + 2]
                    foward_text = strQ2B(foward[1].replace(" ", ""))
                    foward_text = re.sub('＊|\*|\s+|①|②|◎|③|※|④', '', foward_text)
                    parent_text = parent_text + "。" + foward_text
                else:
                    parent_text = parent_text + "。"




        inputs["fname"] = items[8]
        inputs["page_no"] = items[0]

        inputs["text"] = strQ2B(items[1].replace(" ", ""))
        inputs["text"] = re.sub('＊|\*|\s+|①|②|◎|③|※|④', '', inputs["text"])

        inputs["orig_text"] = items[1].replace(" ", "")
        inputs["orig_text"] = re.sub('＊|\*|\s+|①|②|◎|③|※|④', '', inputs["orig_text"])
        
        inputs["index"] = items[2]
        inputs["parent_index"] = int(items[3])
        inputs["parent_text"] = parent_text
        inputs["is_title"] = items[4]
        inputs["is_table"] = items[5]

        if items[6]:
            tags = items[6].split(';')
            values = items[7].split(';')

            if len(tags) == len(values):
                for tag, value in zip(tags, values):
                    value = value.replace(" ", "")
                    value = strQ2B(value)
                    tag = tag.replace(" ", "").replace("＊", "").replace("　", "")
                    inputs["tag"] = table[tag]
                    inputs["value"] = value
                    inputs["start"] = inputs["text"].find(value)

                    clean.append(inputs.copy())

            elif len(tags) > len(values):
                value = values[0]
                value = value.replace(" ", "")
                value = strQ2B(value)
                for tag in tags:
                    tag = tag.replace(" ", "").replace("＊", "").replace("　", "")
                    inputs["tag"] = table[tag]
                    inputs["value"] = value
                    inputs["start"] = inputs["text"].find(value)

                    clean.append(inputs.copy())

            elif len(tags) < len(values):
                tag = tags[0]
                tag = tag.replace(" ", "").replace("＊", "").replace("　", "")
                for value in values:
                    value = value.replace(" ", "")
                    value = strQ2B(value)
                    inputs["tag"] = table[tag]
                    inputs["value"] = value
                    inputs["start"] = inputs["text"].find(value)

                    clean.append(inputs.copy())

        else:
            inputs["tag"] = items[6]
            inputs["value"] = items[7]
            inputs["start"] = -1
            clean.append(inputs.copy())
            

    return clean


def clean_test_data(data, Tag = True):
    clean = []

    length = len(data)

    table = {"調達年度" : 1, "都道府県" : 2, "入札件名" : 3, "施設名" : 4, "需要場所（住所）" : 5, "調達開始日" : 6,
            "調達終了日" : 7, "公告日": 8, "仕様書交付期限" : 9, "質問票締切日時" : 10, "資格申請締切日時" : 11,
            "入札書締切日時" : 12, "開札日時" : 13, "質問箇所所属／担当者" : 14, "質問箇所ＴＥＬ／ＦＡＸ" : 15, "資格申請送付先" : 16,
            "資格申請送付先部署／担当者名" : 17, "入札書送付先" : 18, "入札書送付先部署／担当者名" : 19, "開札場所" : 20}

    for i, items in enumerate(data):
        inputs = {}
        parent_text = ""

        if items[4] == 'x' : items[4] = 1
        if items[5] == 'x' : items[5] = 1

        if items[3]:
            parent = data[int(items[3] - 1)]
            parent_text = "。" + parent[1].replace(" ", "")
            parent_text = re.sub('＊|\*|\s+|①|②|◎|③|※|④', '', parent_text)
            if parent[3]:
                grandparent = data[int(parent[3]) - 1]
                parent_text = grandparent[1].replace(" ", "") + parent_text
                parent_text = re.sub('＊|\*|\s+|①|②|◎|③|※|④', '', parent_text)
                parent_text = strQ2B(parent_text)
            
            if Tag:
                if i - 2 >= 0:
                    previous = data[i - 2]
                    previous_text = strQ2B(previous[1].replace(" ", ""))
                    previous_text = re.sub('＊|\*|\s+|①|②|◎|③|※|④', '', previous_text)
                    parent_text = parent_text + "。" + previous_text
                else:
                    parent_text = parent_text + "。"

                if i - 1 >= 0:
                    previous = data[i - 1]
                    previous_text = strQ2B(previous[1].replace(" ", ""))
                    previous_text = re.sub('＊|\*|\s+|①|②|◎|③|※|④', '', previous_text)
                    parent_text = parent_text + "。" + previous_text
                else:
                    parent_text = parent_text + "。"

                    
                if i + 1 < length:
                    foward = data[i + 1]
                    foward_text = strQ2B(foward[1].replace(" ", ""))
                    foward_text = re.sub('＊|\*|\s+|①|②|◎|③|※|④', '', foward_text)
                    parent_text = parent_text + "。" + foward_text
                else:
                    parent_text = parent_text + "。"

                if i + 2 < length:
                    foward = data[i + 2]
                    foward_text = strQ2B(foward[1].replace(" ", ""))
                    foward_text = re.sub('＊|\*|\s+|①|②|◎|③|※|④', '', foward_text)
                    parent_text = parent_text + "。" + foward_text
                else:
                    parent_text = parent_text + "。"



        inputs["fname"] = items[8]
        inputs["page_no"] = items[0]

        inputs["text"] = strQ2B(items[1].replace(" ", ""))
        inputs["text"] = re.sub('＊|\*|\s+|①|②|◎|③|※|④', '', inputs["text"])

        inputs["orig_text"] = items[1].replace(" ", "")
        inputs["orig_text"] = re.sub('＊|\*|\s+|①|②|◎|③|※|④', '', inputs["orig_text"])

        inputs["index"] = items[2]
        inputs["parent_text"] = parent_text

        clean.append(inputs.copy())
            

    return clean


def strQ2B(ustring):
    """把字串全形轉半形"""
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全形空格直接轉換
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全形字元（除空格）根據關係轉化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)

def read_data(path):
    data = []

    filenames = os.listdir(path)

    for fname in filenames:
        file_path = os.path.join(path, fname)
        process = []

        temp = pd.read_excel(file_path)
        temp["file_name"] = fname[:-9]
        temp = np.array(temp)

        where_is_nan = pd.isnull(temp)
        temp[where_is_nan] = 0
        
        data.extend(temp)

    return data

if __name__ == "__main__":

    path = config.TRAINING_PATH

    data = read_data(path)

    processed_data = clean_data(data)
    # predict_data = clean_test_data(data)

    print(processed_data[1])


    # dataset = QDDataset(predict_data[:4])
    # dataset1 = TagDataset(processed_data[:15])

    # print(dataset[2])
    # print(dataset1[12])
