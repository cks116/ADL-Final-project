import transformers
import torch.nn as nn

class BERTBaseJapanese(nn.Module):
    def __init__(self):
        super(BERTBaseJapanese, self).__init__()

        self.bert = transformers.BertModel.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
        # self.bert.config.type_vocab_size = 2
        self.bert_drop = nn.Dropout(0.2)
        self.linear = nn.Linear(768, 2)
        self.ans = nn.Linear(768, 21)

    def forward(self, index, mask, token_type_ids):
        sequence_output, pooled_output = self.bert(
            index,
            attention_mask = mask,
            token_type_ids = token_type_ids
        )
        # num_tokens = 512
        # sequence output (batch_size, num_tokens, 768)
        # pooled output   (batch_size, 1, 768)
        
        pooled_output = self.bert_drop(pooled_output)
        answer_logits = self.ans(pooled_output).squeeze(-1)

        # sequence_output = self.bert_drop(sequence_output)
        # (batch_size, num_tokens, 768)
        logits = self.linear(sequence_output)
        # (batch_size, num_tokens, 2)
        start_logits, end_logits = logits.split(1, dim = -1)
        # (batch_size, num_tokens, 1), (batch_size, num_tokens, 1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # (batch_size, num_tokens), (batch_size, num_tokens)

        return start_logits, end_logits, answer_logits

