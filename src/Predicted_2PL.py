
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import numpy as np
import copy
from transformers import TrainingArguments, Trainer
import torch
from transformers import DataCollatorWithPadding

torch.cuda.is_available()

# ----------------------------------------------------------------------------------------------------------------------
# FUNCTIONS


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0].flatten()
        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ----------------------------------------------------------------------------------------------------------------------
# Load model, tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
model0 = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-large",
    num_labels=1)  # copy cpu default model, send to GPU

# ----------------------------------------------------------------------------------------------------------------------
# Load  data
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
long = pd.read_csv("data/example_data.csv")
long["key"] = long["key"].astype(int)

# Re-randomize training/eval
np.random.seed(seed=23)
long["data_type"] = "train"
long['rand'] = 0
for item_id in long['item_id'].unique().tolist():
    long.loc[long['item_id'] == item_id, 'rand'] = np.random.rand()
# Need to rank the random number
long.loc[long['rand'].rank() < round(.15 * long.shape[0]), 'data_type'] = "evaluation"
long['data_type'].value_counts()/4
long['data_type'].value_counts()/long.shape[0]

# ----------------------------------------------------------------------------------------------------------------------
# IRT parameters

long['label0'] = long['irt_a']
long['label1'] = -1.7 * long['label0'] * long['irt_b']

# ----------------------------------------------------------------------------------------------------------------------
# Prediction Data
wide = long[['item_id', 'options', 'option', 'data_type', 'key', 'label0', 'label1']].pivot(
    index=['item_id', 'key', 'data_type', 'label0', 'label1'],
    columns='options',
    values='option').reset_index()

# Move correct option to 1st text
t0 = []
t1 = []
t2 = []
t3 = []
for k, v in wide.iterrows():
    key = v['key']
    opts = [0, 1, 2, 3]
    opts.remove(key)
    t0 = t0 + [v[key]]  # t0 is always the correct answer
    t1 = t1 + [v[opts[0]]]
    t2 = t2 + [v[opts[1]]]
    t3 = t3 + [v[opts[2]]]

wide['t0'] = t0
wide['t1'] = t1
wide['t2'] = t2
wide['t3'] = t3
wide['text'] = wide['t0'] + '[SEP]' + wide['t1'] + '[SEP]' + wide['t2'] + '[SEP]' + wide['t3']

# Tokenize
ft_tokenized = tokenizer(wide.text.values.tolist(), truncation=True, max_length=512)
# Dataset
ft_dataset = Dataset(ft_tokenized,
                     # labels are not used so they can be anything
                     [float(0) for i in wide['text'].values.tolist()])
# ----------------------------------------------------------------------------------------------------------------------
# Training/eval Data prep

# Training
t0 = wide.loc[wide.data_type == 'train', 't0']
t1 = wide.loc[wide.data_type == 'train', 't1']
t2 = wide.loc[wide.data_type == 'train', 't2']
t3 = wide.loc[wide.data_type == 'train', 't3']
text0 = (t0 + '[SEP]' + t1 + '[SEP]' + t2 + '[SEP]' + t3).values.tolist()

label0t = wide.loc[wide.data_type == 'train', 'label0'].values.tolist()
label1t = wide.loc[wide.data_type == 'train', 'label1'].values.tolist()

# Eval
te0 = wide.loc[wide.data_type == 'evaluation', 't0']
te1 = wide.loc[wide.data_type == 'evaluation', 't1']
te2 = wide.loc[wide.data_type == 'evaluation', 't2']
te3 = wide.loc[wide.data_type == 'evaluation', 't3']
text_e = (te0 + '[SEP]' + te1 + '[SEP]' + te2 + '[SEP]' + te3).values.tolist()

label0e = wide.loc[wide.data_type == 'evaluation', 'label0'].values.tolist()
label1e = wide.loc[wide.data_type == 'evaluation', 'label1'].values.tolist()

# ----------------------------------------------------------------------------------------------------------------------
# Loop IRT a, d, and repeat 1 and 4

labs = ['irt_a', 'irt_d']
out_fit = pd.DataFrame()
out_loss = pd.DataFrame()
lab = 'irt_d'

for lab in labs:
    model = copy.deepcopy(model0).to("cuda")
    torch.device("cuda")

    if lab == 'irt_a':
        label_t = label0t
        label_e = label0e
    if lab == 'irt_d':
        label_t = label1t
        label_e = label1e

    # ----------------------------------------------------------------------------------------------------------------------
    # Training Data
    tokenized_eval = tokenizer(text_e,
                               truncation=True,
                               max_length=512)
    tokenized_train = tokenizer(text0, truncation=True, max_length=512)
    train_dataset = Dataset(tokenized_train, label_t)

    # ----------------------------------------------------------------------------------------------------------------------
    # Eval
    eval_dataset = Dataset(tokenized_eval, label_e)

    # ----------------------------------------------------------------------------------------------------------------------
    # FINE-TUNE

    training_args = TrainingArguments(
        output_dir="./output/saved_models",
        optim="adamw_torch",
        learning_rate=1e-6,
        per_device_train_batch_size=4,  # lower batch size if memory runs out.
        per_device_eval_batch_size=4,
        num_train_epochs=2,
        weight_decay=.1,
        evaluation_strategy="epoch",
        # save_strategy="epoch",
        # load_best_model_at_end=True, # this loads the best model out of all epochs. So every epoch must be saved
        # save_total_limit=1,
        # logging_strategy='steps',
        # logging_steps=20,
        # gradient_accumulation_steps = 4,
        # gradient_checkpointing = True,
        fp16=False,  # use if memory runs out. This uses only 4 decimal points
    )
    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()

    # ------------------------------------------------------------------------------------------------------------------
    # PREDICT WITH EVALUATION DATA

    pred = trainer.predict(ft_dataset).predictions
    out_fit0 = wide[['item_id', 'data_type']].copy()
    out_fit0["lab"] = lab
    out_fit0["pred"] = pred.flatten().tolist()
    out_fit = pd.concat([out_fit, out_fit0])

    # ------------------------------------------------------------------------------------------------------------------
    # Model Loss

    out_loss0 = pd.DataFrame(trainer.state.log_history)
    out_loss0["lab"] = lab
    out_loss = pd.concat([out_loss, out_loss0])

    del model
    del trainer
    torch.cuda.empty_cache()

# ----------------------------------------------------------------------------------------------------------------------
# SAVE DATA
out_fit.to_csv(path_or_buf="output/Predicted_2PL_out_fit.csv", index=False)
out_loss.to_csv(path_or_buf="output/Predicted_2PL_out_loss.csv", index=False)
























