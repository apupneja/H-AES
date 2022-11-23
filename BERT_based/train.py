import torch
import pandas as pd
import config
from sklearn import model_selection
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup

from qwk import quadratic_weighted_kappa
from dataset import BERTDataset
from model import BERTBase
from engine import train_fn, eval_fn

test_qwk = []
train_qwk = []
val_qwk = []


def eval(outputs, targets, min_df_train, div_factor, CURR):
    outputs = [round(i * div_factor + min_df_train) for i in outputs]
    targets = [round(i) for i in targets]

    qwk = quadratic_weighted_kappa(targets, outputs)
    print(f"qwk = {qwk}")
    if CURR == 'train':
        train_qwk.append(qwk)
    elif CURR == 'test':
        test_qwk.append(qwk)
    elif CURR == 'validation':
        val_qwk.append(qwk)


def draw_plot():
    x = [i for i in range(config.EPOCHS)]
    y1 = train_qwk
    y2 = val_qwk

    plt.plot(x, y1, label='Train QWK')
    plt.plot(x, y2, label='Val QWK')
    plt.legend()
    plt.show()


def training_loop(train_data_loader=None
                  , optimizer=None
                  , model=None
                  , device=None
                  , scheduler=None
                  , valid_data_loader=None
                  , div_factor=None
                  , min_df_train=None
                  ):
    for epoch in range(config.EPOCHS):
        print(f"running epoch = {epoch}")
        train_fn(train_data_loader, model, optimizer, device, scheduler)
        train_outputs, train_targets = eval_fn(train_data_loader, model, device)
        outputs, targets = eval_fn(valid_data_loader, model, device)
        train_targets = [round(i * div_factor + min_df_train) for i in train_targets]
        eval(train_outputs, train_targets, min_df_train, div_factor, 'train')
        eval(outputs, targets, min_df_train, div_factor, 'validation')


def run():
    df = pd.read_excel(config.DATASET).dropna()

    input_df = df

    df_train, df_valid = model_selection.train_test_split(
        input_df, test_size=0.2, random_state=42
        , stratify=df.Score.values
    )

    df_test, df_valid = model_selection.train_test_split(
        df_valid, test_size=0.5, random_state=42
        , stratify=df_valid.Score.values
    )

    df_test = df_test.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    col = "Score"

    min_df_train = input_df[col].min()
    div_factor = input_df[col].max() - input_df[col].min()
    df_train[col] = (df_train[col] - min_df_train) / div_factor

    train_dataset = BERTDataset(
        essay_hindi=df_train.essay_hindi.values, score=df_train.score.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = BERTDataset(
        essay_hindi=df_valid.essay_hindi.values, score=df_valid.score.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    test_dataset = BERTDataset(
        essay_hindi=df_test.essay_hindi.values, score=df_test.score.values
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device(config.DEVICE)
    model = BERTBase()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)

    optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    training_loop(train_data_loader, optimizer, model, device, scheduler, valid_data_loader, div_factor, min_df_train)
    outputs, targets = eval_fn(test_data_loader, model, device)
    eval(outputs, targets, min_df_train, div_factor, 'test')

    draw_plot()


if __name__ == "__main__":
    run()
