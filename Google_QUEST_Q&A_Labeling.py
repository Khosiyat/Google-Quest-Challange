#INSTALL THE FOLLOWINGS FOR CUDA
# conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# print(torch.__version__)
# my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device="cpu")
# print(my_tensor)
# torch.cuda.is_available()

# conda activate --stack myenv

#LOAD THE FOLLOWING DATASETS FROM KAGGLE
#1) ../input/google-quest-challenge
#2)../input/bert-base-uncased
#3)../input/tpubert

import os
import torch
import pandas as pd
from scipy import stats
import numpy as np

from tqdm import tqdm
import torch.nn as nn
import joblib

import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
import sys


class Uncased_BERTBase(nn.Module):
    def __init__(self, path_transformer):
        super(Uncased_BERTBase, self).__init__()
        self.path_transformer = path_transformer
        self.bert = transformers.BertModel.from_pretrained(self.path_transformer)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 30)

    def forward_step(self, _id, attentionMASK, tokenType_id):
        _, o2 = self.bert(_id, attentionMASK=attentionMASK, tokenType_id=tokenType_id)
        bert_o2 = self.bert_drop(o2)
        bert_out_o2 = self.out(bert_o2)
        return bert_out_o2


class BERT_DatasetTest:
    def __init__(self, titleQuestion, questionBert_o2dy, replyBert, tokenizerBert, max_length):
        self.titleQuestion = titleQuestion
        self.questionBert_o2dy = questionBert_o2dy
        self.replyBert = replyBert
        self.tokenizerBert = tokenizerBert
        self.max_length = max_length

    def __len__(self):
        return len(self.replyBert)

    def __getitem__(self, item_unit):
        question_title = str(self.titleQuestion[item_unit])
        question_bert_o2dy = str(self.questionBert_o2dy[item_unit])
        replyBert_text = str(self.replyBert[item_unit])

        question_title = " ".join(question_title.split())
        question_bert_o2dy = " ".join(question_bert_o2dy.split())
        replyBert_text = " ".join(replyBert_text.split())

        inputs = self.tokenizerBert.encode_plus(
            question_title + " " + question_bert_o2dy,
            replyBert_text,
            add_specialTokens=True,
            max_length=self.max_length,
        )
        _id = inputs["input__id"]
        tokenType_id = inputs["tokenType_id"]
        attentionMASK = inputs["attentionMASK"]
        
        padding_length = self.max_length - len(_id)
        
        _id = _id + ([0] * padding_length)
        attentionMASK = attentionMASK + ([0] * padding_length)
        tokenType_id = tokenType_id + ([0] * padding_length)
        
        return {
            '_id': torch.tensor(_id, dtype=torch.long),
            'attentionMASK': torch.tensor(attentionMASK, dtype=torch.long),
            'tokenType_id': torch.tensor(tokenType_id, dtype=torch.long)
        }


def predict():
    DEVICE_CUDA =torch.device("cuda")
    BATCH_SIZE_test = 8
    DATASET_test = "../input/google-quest-challenge/test.csv"
    dataFrame = pd.read_csv(DATASET_test).fillna("none")

    titleQuestion = dataFrame.question_title.values.astype(str).tolist()
    questionBert_o2dy = dataFrame.question_body.values.astype(str).tolist()
    replyBert = dataFrame.answer.values.astype(str).tolist()
    category_strType = dataFrame.category.values.astype(str).tolist()

    tokenizerBert = transformers.BertTokenizer.from_pretrained("../input/bert-base-uncased", 
                                                           do_lower_case=True)
    maxlen = 512
    predictions = []

    DATASET_test = BERT_DatasetTest(
        titleQuestion=titleQuestion,
        questionBert_o2dy=questionBert_o2dy,
        replyBert=replyBert,
        tokenizerBert=tokenizerBert,
        max_length=maxlen
    )
    test_data_loader = torch.utils.data.DataLoader(
        DATASET_test,
        batch_size=BATCH_SIZE_test,
        shuffle=False,
        num_workers=4
    )

    modelBert = Uncased_BERTBase("../input/bert-base-uncased/")
    modelBert.to(DEVICE_CUDA)
    modelBert.load_state_dict(torch.load("../input/tpubert/model.bin"))
    modelBert.eval()

    tqdm_progress = tqdm(test_data_loader, total=int(len(DATASET_test) / test_data_loader.batch_size))
    for tqdm_ in enumerate(tqdm_progress):
        _id = tqdm_["_id"]
        attentionMASK = tqdm_["attentionMASK"]
        tokenType_id = tqdm_["tokenType_id"]

        _id = _id.to(DEVICE_CUDA, dtype=torch.long)
        attentionMASK = attentionMASK.to(DEVICE_CUDA, dtype=torch.long)
        tokenType_id = tokenType_id.to(DEVICE_CUDA, dtype=torch.long)
        
        with torch.no_grad():
            outputs = modelBert(
                _id=_id,
                attentionMASK=attentionMASK,
                tokenType_id=tokenType_id
            )
            outputs = torch.sigmoid(outputs).cpu().numpy()
            predictions.append(outputs)

    return np.vstack(predictions)



predictions = predict()


SUBMISSION = "../input/google-quest-challenge/SUBMISSION.csv"
submit_myWork = pd.read_csv(SUBMISSION)
target_cols = list(submit_myWork.drop("qa_id", axis=1).columns)

submit_myWork[target_cols] = predictions

submit_myWork.head()

submit_myWork.to_csv("submission.csv", index=False)

#Credits to Abhishek Thakur for his great work
