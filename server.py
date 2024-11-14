import fastapi
from pydantic import BaseModel
import torch
import uvicorn
from transformers import (ElectraTokenizerFast, ElectraForSequenceClassification, BertTokenizerFast, BertForTokenClassification, pipeline)

emotion_ids = ["rage", "sadness", "happiness", "anxiety"] # 변경 X
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
criteria_prob = 0.6

emotion_model_name = "./model"
emotion_tokenizer_name = "monologg/koelectra-small-v3-discriminator"
emotion_tokenizer = ElectraTokenizerFast.from_pretrained(emotion_tokenizer_name)
emotion_model = ElectraForSequenceClassification.from_pretrained(emotion_model_name)


ner_model_name = 'joon09/kor-naver-ner-name'
ner_tokenizer = BertTokenizerFast.from_pretrained(ner_model_name)
ner_model = BertForTokenClassification.from_pretrained(ner_model_name)
pipe = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, device=device)

class DocumentContent(BaseModel):
    content: str

app = fastapi.FastAPI()

@app.post("/emotions")
async def get_emotions(content: DocumentContent):
    sequences = content.content.split('.')
    sequences_total_len = len(sequences)

    emotion_count = [0,0,0,0]

    for seq in sequences:
        ids = emotion_tokenizer(seq, return_tensors="pt").to(device)
        logit = emotion_model(**ids)

        probs = torch.sigmoid(logit)

        for i, prob in enumerate(probs[0]):
            if prob > criteria_prob: 
                emotion_count[i] += 1
    
    emotion_count /= sequences_total_len

    res = {}

    for i in range(4):
        res[emotion_ids[i]] = emotion_count[i]

    return res

@app.post("/ner")
async def censor_name(content: DocumentContent):
    seqs = content.content.split('.')

    for i, seq in enumerate(seqs):
        results = pipe(seq, grouped_entities=True, aggregation_strategy='average')
        if len(results) == 0: continue

        tmp = list(seq)
        for res in results:
            
            for j in range(res['start'],res['end']):
                tmp[j] = '*'
            
        seqs[i] = ''.join(tmp)

    return '.'.join(seqs)
        
if __name__ == '__main__':
    uvicorn.run("server:app", host="0.0.0.0", port=8080)
