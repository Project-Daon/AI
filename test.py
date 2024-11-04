import torch
from transformers import (ElectraTokenizerFast, ElectraForSequenceClassification)

model_name = "./model"
tokenizer_name = "monologg/koelectra-small-v3-discriminator"
tokenizer = ElectraTokenizerFast.from_pretrained(tokenizer_name)
model = ElectraForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

criteria = 0.5

model.to(device)

emotion_ids = ["분노", "슬픔", "기쁨", "불안"]


def predict(text: str) -> str:
    input = tokenizer(text, return_tensors="pt").to(device)

    logit = model(**input)

    return torch.sigmoid(logit.logits)

if __name__ == "__main__":
    while True:
        output = predict(input("Enter Sentence >> "))
        result = []
        for i, x in enumerate(output[0]):
            if x > criteria:
                result.append(emotion_ids[i])
        print(result)
