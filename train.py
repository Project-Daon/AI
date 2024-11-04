import time
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import (Trainer, TrainingArguments, ElectraForSequenceClassification, ElectraTokenizerFast, EarlyStoppingCallback)


model_name = "monologg/koelectra-small-v3-discriminator"
epochs = 20
batch_size = 32
train_dataset_path = "data/training_data.csv"
valid_dataset_path = "data/validation_data.csv"
output_dir = "output/" + time.strftime("%Y%m%d-%H%M%S")
emotion_ids = {"분노":[1,0,0,0], "슬픔":[0,1,0,0], "기쁨":[0,0,1,0], "불안":[0,0,0,1]}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.datas.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
    
    def __len__(self):
        return len(self.labels)
    
def compute_metrics(pred):
    labels = pred.label_ids.argmax(-1)
    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')

    return {"accuracy": acc, "f1":f1}

    
tokenizer = ElectraTokenizerFast.from_pretrained(model_name)

raw_train_dataset = pd.read_csv(train_dataset_path)
raw_valid_dataset = pd.read_csv(valid_dataset_path)

x_train = tokenizer(raw_train_dataset["사람문장1"].values.tolist(), truncation=True, padding=True)
y_train = [emotion_ids[x] for x in raw_train_dataset["감정_대분류"].values.tolist()]

x_valid = tokenizer(raw_valid_dataset["사람문장1"].values.tolist(), truncation=True, padding=True)
y_valid = [emotion_ids[x] for x in raw_valid_dataset["감정_대분류"].values.tolist()]

train_dataset = Dataset(x_train, y_train)
valid_datset = Dataset(x_valid, y_valid)

train_args = TrainingArguments(output_dir=output_dir,
                               num_train_epochs=epochs,
                               per_device_train_batch_size=batch_size,
                               per_device_eval_batch_size=batch_size,
                               warmup_steps=500,
                               weight_decay=0.01,
                               logging_dir=output_dir,
                               load_best_model_at_end=True,
                               optim="adamw_torch",
                               fp16=True,
                               eval_strategy="epoch",
                               save_strategy="epoch"
                               )

model = ElectraForSequenceClassification.from_pretrained(model_name, num_labels=4, problem_type="multi_label_classification")

trainer = Trainer(model=model, args=train_args, 
                  train_dataset=train_dataset, 
                  eval_dataset=valid_datset,
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
                  compute_metrics=compute_metrics)

trainer.train()

trainer.save_model(output_dir+"/model")
