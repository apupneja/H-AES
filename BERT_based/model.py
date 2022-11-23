import torch.nn as nn
import config


class BERTBase(nn.Module):
    def __init__(self):
        super(BERTBase, self).__init__()
        self.bert = config.MODEL
        self.bert_drop = nn.Dropout(0.3)

        self.dense_layer_1 = nn.Linear(768, 512)

        self.dense_layer_2 = nn.Linear(512, 128)
        self.dense_layer_3 = nn.Linear(128, 64)

        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.25)

        self.out = nn.Linear(64, 1)
        self.sig = nn.Sigmoid()

    def forward(self, ids, mask):
        _, o2 = self.bert(ids, attention_mask=mask, return_dict=False)

        dense_output_1 = self.dense_layer_1(o2)
        relu_output_1 = self.act(dense_output_1)
        d1 = self.drop(relu_output_1)
        dense_output_2 = self.dense_layer_2(d1)
        relu_output_2 = self.act(dense_output_2)
        d2 = self.drop(relu_output_2)

        dense_output_3 = self.dense_layer_3(d2)
        relu_output_3 = self.act(dense_output_3)
        d3 = relu_output_3
        output = self.out(d3)

        return output
