from transformers import BertTokenizer

MODEL_PATH = "try"
TAG_MODEL_PATH = "TagBert.bin"
# TAG_MODEL_PATH = "try_19.bin"
# TAG_MODEL_PATH = "new.bin"
PREDICT_MODEL_PATH = "PredictBert.bin"
# PREDICT_MODEL_PATH = "try_4.bin"


MAX_LEN = 512
TRAIN_BATCH_SIZE = 6
VALID_BATCH_SIZE = 16
EPOCHS = 20

TRAINING_PATH = "./data/train/ca_data"
VALIDATE_PATH = "./data/dev/ca_data"
TESTING_PATH = "./data/test/ca_data"

TOKENIZER = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
# tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")


if __name__ == "__main__":
    data =  TOKENIZER.encode("外部公開システムにおけるセキュリティ監視業務")
    # data1 = tokenizer.encode("外部公開システムにおけるセキュリティ監視業務")

    # print(len("。入札公告"))

    print(data)
    print(TOKENIZER.decode(data))
