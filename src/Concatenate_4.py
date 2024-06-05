
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


class Dataset_mult(torch.utils.data.Dataset):
    def __init__(self, encodings, labels0, labels1, labels2, labels3):
        self.encodings = encodings
        self.labels0 = labels0
        self.labels1 = labels1
        self.labels2 = labels2
        self.labels3 = labels3
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor([self.labels0[idx], self.labels1[idx], self.labels2[idx], self.labels3[idx]])
        return item
    def __len__(self):
        return len(self.labels0)


class RegressionTrainerMult_CEL(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        predictions = outputs[0]
        predictions = torch.nn.functional.softmax(predictions, dim=1)
        CEL = -(labels * torch.log(predictions) + (1 - labels) * torch.log(1 - predictions))
        loss = torch.sum(CEL)
        return (loss, outputs) if return_outputs else loss


# ----------------------------------------------------------------------------------------------------------------------
# Load model, tokenizer
model0 = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-large",
    num_labels=4)
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

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
# Prediction Data
wide = long[['item_id', 'options', 'option', 'data_type', 'key']].pivot(index=['item_id', 'key', 'data_type'],
                                                                        columns='options',
                                                                        values='option').reset_index()
wide['text'] = wide[0] + '[SEP]' + wide[1] + '[SEP]' + wide[2] + '[SEP]' + wide[3]
# ft_data = copy.deepcopy(long)
# Tokenize
ft_tokenized = tokenizer(wide.text.values.tolist(), truncation=True, max_length=512)
# Dataset
ft_dataset = Dataset_mult(ft_tokenized,
                             # labels are not used so they can be anything
                             [float(0) for i in wide['text'].values.tolist()],
                             [float(0) for i in wide["text"].values.tolist()],
                             [float(0) for i in wide["text"].values.tolist()],
                             [float(0) for i in wide["text"].values.tolist()])

# ----------------------------------------------------------------------------------------------------------------------
# Training Data (only text)
t0 = wide.loc[wide.data_type == 'train', 0]
t1 = wide.loc[wide.data_type == 'train', 1]
t2 = wide.loc[wide.data_type == 'train', 2]
t3 = wide.loc[wide.data_type == 'train', 3]
text0 = (t0 + '[SEP]' + t1 + '[SEP]' + t2 + '[SEP]' + t3).values.tolist()
text1 = (t2 + '[SEP]' + t3 + '[SEP]' + t0 + '[SEP]' + t1).values.tolist()
text2 = (t1 + '[SEP]' + t0 + '[SEP]' + t3 + '[SEP]' + t2).values.tolist()
text3 = (t3 + '[SEP]' + t2 + '[SEP]' + t1 + '[SEP]' + t0).values.tolist()

# Eval
te0 = wide.loc[wide.data_type == 'evaluation', 0]
te1 = wide.loc[wide.data_type == 'evaluation', 1]
te2 = wide.loc[wide.data_type == 'evaluation', 2]
te3 = wide.loc[wide.data_type == 'evaluation', 3]
text_e = (te0 + '[SEP]' + te1 + '[SEP]' + te2 + '[SEP]' + te3).values.tolist()
tokenized_eval = tokenizer(text_e,
                           truncation=True,
                           max_length=512)

# ----------------------------------------------------------------------------------------------------------------------
# BEGIN MODEL_ID LOOP
# Loop once per theta
thetas = -np.sort(-np.arange(-3, 3.1, .1))
out_fit = pd.DataFrame()
out_loss = pd.DataFrame()
model_id = int(0)
# ------------------------------------------------------------------------------------------------------------------
# Load model
model = copy.deepcopy(model0).to("cuda") # send to gpu
torch.device("cuda")
# ----------------------------------------------------------------------------------------------------------------------
# Training Text Data
tokenized_train = tokenizer(text0 + text1 + text2 + text3,
                            truncation=True,
                            max_length=512)
for theta in thetas:
    # break
    # --------------------------------------------------------------------------------------------------------------
    # Calculate selection probabilities
    # loop to simulate data
    # Simulate responses on both train and eval data.
    model_id = model_id + 1  # add 1 to model_id
    print("Begin")
    print(model_id)
    print(theta)
    resp_df = pd.DataFrame()

    for item_id in long.item_id.unique():
    # for item_id in long.loc[long.data_type == 'train', 'item_id'].unique():
        # get single item data. Set options as the index
        long1 = long.loc[long.item_id == item_id].set_index('options').copy()
        option = long1[['item_id', 'option', 'data_type', 'key']]
        aa = long1['irt_a'].tolist()[0]
        bb = long1['irt_b'].tolist()[0]
        pc = long1["prp_choosing"]
        key = long1['key'][0]
        # calculate p
        Z = 1.7*aa*(theta-bb)
        p = np.exp(Z)/(1+np.exp(Z))
        distractors = pc[pc.index != key]
        distractors_p = ((1 - p) * distractors / sum(distractors)).tolist()
        # Save Data
        df = pd.DataFrame({
            'model_id': model_id,
            'theta': theta,
            'p': [p] + distractors_p,
            'options': [key] + option.index[option.index != key].tolist()
        })
        # num row = number of options
        df2 = pd.merge(option.reset_index(), df, on='options')
        resp_df = pd.concat([resp_df, df2])
    resp_df = resp_df.reset_index()

    # ----------------------------------------------------------------------------------------------------------------------
    # CLEAN DATA
    # 85% training, 15% evaluating
    # Train data: the anchors with real student response data and item parameters.
    # Evaluation data: field-test data.

    # Training
    train0 = resp_df.loc[resp_df.data_type == 'train', ['item_id', 'options', 'option', 'p']].copy()
    train = train0.pivot(index='item_id', columns='options', values=['option', 'p'])
    # dataset
    p0 = [float(i) for i in train.p[0].values.tolist()]
    p1 = [float(i) for i in train.p[1].values.tolist()]
    p2 = [float(i) for i in train.p[2].values.tolist()]
    p3 = [float(i) for i in train.p[3].values.tolist()]

    train_dataset = Dataset_mult(tokenized_train,
                                    p0 + p2 + p1 + p3,
                                    p1 + p3 + p0 + p2,
                                    p2 + p0 + p3 + p1,
                                    p3 + p1 + p2 + p0)

    # Eval
    evaluation0 = resp_df.loc[resp_df.data_type == 'evaluation', ['item_id', 'options', 'option', 'p']].copy()
    evaluation = evaluation0.pivot(index='item_id', columns='options', values=['option', 'p'])
    # Dataset
    eval_dataset = Dataset_mult(tokenized_eval,
                                   [float(i) for i in evaluation.p[0].values.tolist()],
                                   [float(i) for i in evaluation.p[1].values.tolist()],
                                   [float(i) for i in evaluation.p[2].values.tolist()],
                                   [float(i) for i in evaluation.p[3].values.tolist()])

    # ----------------------------------------------------------------------------------------------------------------------
    # FINE-TUNE
    # Initial Training for the "smartest" AI
    if model_id == 1:
        training_args = TrainingArguments(
            output_dir="./output/saved_models",
            optim="adamw_torch",
            learning_rate=7e-6,
            per_device_train_batch_size=4,  # lower batch size if memory runs out.
            per_device_eval_batch_size=4,
            num_train_epochs=15, 
            weight_decay=0.2,
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
        trainer = RegressionTrainerMult_CEL(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        trainer.train()
        out_loss0 = pd.DataFrame(trainer.state.log_history)
        out_loss0["theta"] = theta
        out_loss0["model_id"] = model_id
        out_loss = pd.concat([out_loss, out_loss0])

    # General Train
    training_args = TrainingArguments(
        output_dir="./output/saved_models",
        optim="adamw_torch",
        learning_rate=1e-6,
        per_device_train_batch_size=4,  # lower batch size if memory runs out.
        per_device_eval_batch_size=4,
        num_train_epochs=2,
        weight_decay=.2,
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
    trainer = RegressionTrainerMult_CEL(
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
    ft_data1 = copy.deepcopy(wide.reset_index()[['item_id', 'data_type', 'key']])
    ft_data1["theta"] = theta
    ft_data1["model_id"] = model_id
    ft_data1["p0"] = [np.exp(i[0])/sum(np.exp(i)) for i in pred]
    ft_data1["p1"] = [np.exp(i[1])/sum(np.exp(i)) for i in pred]
    ft_data1["p2"] = [np.exp(i[2])/sum(np.exp(i)) for i in pred]
    ft_data1["p3"] = [np.exp(i[3])/sum(np.exp(i)) for i in pred]
    ft_data1 = ft_data1.sort_values(by=['data_type', 'key'])  # Sort
    out_fit = pd.concat([out_fit, ft_data1])

    # ------------------------------------------------------------------------------------------------------------------
    # Model Fit, probability Data
    # Keep this data minimal
    out_loss0 = pd.DataFrame(trainer.state.log_history)
    out_loss0["theta"] = theta
    out_loss0["model_id"] = model_id
    out_loss = pd.concat([out_loss, out_loss0])

    # ------------------------------------------------------------------------------------------------------------------
    # END THETA LOOP

# ----------------------------------------------------------------------------------------------------------------------
# SAVE DATA
out_fit.to_csv(path_or_buf="output/Concatenate_4_out_fit.csv", index=False)
out_loss.to_csv(path_or_buf="output/Concatenate_4_out_loss.csv", index=False)








