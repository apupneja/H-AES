import config
import torch


class BERTDataset:
    def __init__(self, essay_hindi, score):
        self.essay_hindi = essay_hindi
        self.score = score
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.essay_hindi)

    def __getitem__(self, item):
        essay_hindi = str(self.essay_hindi[item])
        essay_hindi = " ".join(essay_hindi.split())

        inputs = self.tokenizer.encode_plus(
            config.CURR_PROMPT,
            essay_hindi,
            add_special_tokens=True,
            truncation='only_second',
            max_length=self.max_len,
            padding='max_length',
        )
        # print(inputs)
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "scores": torch.tensor(self.score[item], dtype=torch.float),
        }
