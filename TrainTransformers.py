#!/usr/bin/env python3
"""
> Run the training script:
    > python TrainTransformers.py --model_name="camembert-base"
Authors
 * Yanis LABRAK 2023
"""

import os
import uuid
import argparse

import torch
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, accuracy_score, classification_report

from datasets import load_dataset, load_metric

import transformers
from transformers import AutoTokenizer, EvalPrediction, AutoModelForSequenceClassification, TrainingArguments, Trainer

print(transformers.__version__)

parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("-m", "--model_name", help = "HuggingFace Hub model name")
args = vars(parser.parse_args())

THRESHOLD_VALUE = 0.70

dataset_base = load_dataset(
    'csv',
    data_files={
        'train': f"./train.tsv",
        'validation': f"./dev.tsv",
        'test': f"./test.tsv",
    },
    delimiter="\t",
)

dataset_train = dataset_base["train"]
dataset_val = dataset_base["validation"]
dataset_test = dataset_base["test"]

metric = load_metric("accuracy")

labels_list = []
for element in dataset_train:
    labels_list.extend(element["subheading"].split("|"))
labels_list = sorted(list(set(labels_list)))

num_labels = len(labels_list)

batch_size = 12
EPOCHS = 32

model_checkpoint = str(args["model_name"])

id2label = {idx:label for idx, label in enumerate(labels_list)}
label2id = {label:idx for idx, label in enumerate(labels_list)}

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    problem_type="multi_label_classification",
    num_labels=num_labels,
)

model_name = model_checkpoint.split("/")[-1]

def preprocess_function(e):
    res = tokenizer(e["abstract"], truncation=True, max_length=512, padding="max_length")
    labels = [0.0] * len(labels_list)
    for pub_type in e["subheading"].split("|"):
        pub_type = label2id[pub_type]
        labels[pub_type] = 1.0
    res["labels"] = labels
    return res

dataset_train = dataset_train.map(preprocess_function, batched=False)
dataset_train.set_format("torch")

dataset_val   = dataset_val.map(preprocess_function, batched=False)
dataset_val.set_format("torch")

dataset_test   = dataset_test.map(preprocess_function, batched=False)
ids_test = list(dataset_test["identifier"])
true_labels = list(dataset_test["labels"])
dataset_test.set_format("torch")

output_dir = f"./models/{model_name}-finetuned-{uuid.uuid4().hex}"

training_args = TrainingArguments(
    output_dir,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    push_to_hub=False,
    greater_is_better=True,
    metric_for_best_model="f1_weighted",
)

def toLogits(predictions, threshold=THRESHOLD_VALUE):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    return y_pred

def multi_label_metrics(predictions, labels, threshold=THRESHOLD_VALUE):
    y_pred = toLogits(predictions, threshold)
    y_true = labels
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_weighted_average = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {'f1_macro': f1_macro_average, 'f1_micro': f1_micro_average,  'f1_weighted': f1_weighted_average, 'accuracy': accuracy, 'roc': roc_auc}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids
    )
    return result

trainer = Trainer(
    model,
    training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

best_model_path = f"{output_dir}/best_model/"
trainer.save_model(best_model_path)
print(f"Best model is saved at : {best_model_path}")

print(trainer.evaluate())

# ------------------ EVALUATION ------------------

predictions, labels, _ = trainer.predict(dataset_test)
predictions = toLogits(predictions, THRESHOLD_VALUE)

metrics = multi_label_metrics(predictions, true_labels, THRESHOLD_VALUE)

cr = classification_report(true_labels, predictions, target_names=labels_list, digits=4)
print(cr)

def toLetters(predictions):
        
    predictions_classes = []

    for p in predictions:
        predictions_classes.append(
            [id2label[i] for i, p in enumerate(p) if p == 1]
        )
    
    return predictions_classes

predictions_letters = toLetters(predictions)

os.makedirs("./submissions/", exist_ok=True)
f_out_submission = open(f"./submissions/submission-{uuid.uuid4().hex}.txt","w")
for indentifier, pred_value in zip(ids_test, predictions_letters):
    pred_value = "|".join(pred_value)
    f_out_submission.write(f"{indentifier};{pred_value}\n")
f_out_submission.close()
