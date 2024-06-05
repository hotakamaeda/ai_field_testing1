
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

class Dataset_vert(torch.utils.data.Dataset):
    def __init__(self, encodings, item_id, labels):
        self.encodings = encodings
        self.labels = labels
        self.item_id = item_id
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor([self.labels[idx]])
        item['item_id'] = torch.tensor([self.item_id[idx]])
        return item
    def __len__(self):
        return len(self.labels)


class RegressionTrainerMult_CEL_vert(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").reshape([-1, 4])
        outputs = model(**inputs)
        predictions0 = outputs[0][:, 0]
        # separete list by item id (by 4)
        predictions = predictions0.reshape([-1, 4])
        predictions_s = torch.nn.functional.softmax(predictions, dim=1)
        CEL = -(labels * torch.log(predictions_s) + (1 - labels) * torch.log(1 - predictions_s))
        loss = torch.sum(CEL)
        return (loss, outputs) if return_outputs else loss


# ----------------------------------------------------------------------------------------------------------------------
# Load model, tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-large",
    num_labels=1).to("cuda")  # copy cpu default model, send to GPU
torch.device("cuda")

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
# Tokenize
ft_tokenized = tokenizer(long.option.values.tolist(), truncation=True, max_length=512)
# Dataset
ft_dataset = Dataset_vert(ft_tokenized,
                         # labels are not used so they can be anything
                         [int(i) for i in long['item_id'].values.tolist()],
                         [float(0) for i in long['option'].values.tolist()])

# ----------------------------------------------------------------------------------------------------------------------
# BEGIN MODEL_ID LOOP
# Loop once per theta
thetas = -np.sort(-np.arange(-3, 3.1, .1))
model_id = int(0)
out_fit = pd.DataFrame()
out_loss = pd.DataFrame()
for theta in thetas:
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
    # Tokenize, create dataset

    # Training
    train = resp_df.loc[resp_df.data_type == 'train', ].copy()
    tokenized_train = tokenizer(train.option.values.tolist(),
                                truncation=True,
                                max_length=512)
    train_dataset = Dataset_vert(tokenized_train, train.item_id.values.tolist(), train.p.values.tolist())

    # Eval
    evaluation = resp_df.loc[resp_df.data_type == 'evaluation', ].copy()
    tokenized_eval = tokenizer(evaluation.option.values.tolist(),
                               truncation=True,
                               max_length=512)
    eval_dataset = Dataset_vert(tokenized_eval, evaluation.item_id.values.tolist(), evaluation.p.values.tolist())

    # ----------------------------------------------------------------------------------------------------------------------
    # FINE-TUNE

    # Initial Training for the "smartest" AI
    if model_id == 1:
        training_args = TrainingArguments(
            output_dir="./output/saved_models",
            optim="adamw_torch",
            learning_rate=4e-6,
            per_device_train_batch_size=16,  # lower batch size if memory runs out.
            per_device_eval_batch_size=16,
            num_train_epochs=10,
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
        trainer = RegressionTrainerMult_CEL_vert(
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
        learning_rate=1e-6, # probably lower
        per_device_train_batch_size=16,  # lower batch size if memory runs out.
        per_device_eval_batch_size=16,
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
    trainer = RegressionTrainerMult_CEL_vert(
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
    pred2 = pred.reshape([-1, 4])
    pred2_s = [np.exp(i) / sum(np.exp(i)) for i in pred2]
    ft_data1 = copy.deepcopy(long.reset_index()[['item_id', 'data_type', 'key', 'options']])
    ft_data1["theta"] = theta
    ft_data1["model_id"] = model_id
    ft_data1["pred"] = np.array(pred2_s).flatten().tolist()
    ft_data1 = ft_data1.sort_values(by=['data_type', 'key', 'item_id', 'options'])  # Sort
    out_fit = pd.concat([out_fit, ft_data1])

    # ------------------------------------------------------------------------------------------------------------------
    # Model Fit, probability Data
    out_loss0 = pd.DataFrame(trainer.state.log_history)
    out_loss0["theta"] = theta
    out_loss0["model_id"] = model_id
    out_loss = pd.concat([out_loss, out_loss0])

    # ------------------------------------------------------------------------------------------------------------------
    # END THETA LOOP


# ----------------------------------------------------------------------------------------------------------------------
# SAVE DATA
out_fit.to_csv(path_or_buf="output/Separate_out_fit.csv", index=False)
out_loss.to_csv(path_or_buf="output/Separate_out_loss.csv", index=False)
























