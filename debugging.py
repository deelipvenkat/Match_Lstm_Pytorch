import pandas as pd
import numpy as np
import pickle
import yaml
from time import perf_counter
import gc
import sys

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast , GradScaler

from preprocessing import splitter,train_preprocessor
from models import load_model
from utils import get_answer_index
from dataloader import batched_data
from evaluation import proba_score,answer_from_context,tokens_to_sentence,validate,validate_f1,validation_loss
from train import trainer
from metrics import f1_score , em_score
from dataloader import batched_data,batched_indexed_data,squad_data,squad_index_data


if __name__=='__main__':
  
  with open('/home/config.yaml') as f:
    config=yaml.safe_load(f)

  device=['cuda' if torch.cuda.is_available() is True else 'cpu'][0]

  f=open('/home/pickle_objects/squad_weight_2.pkl','rb')
  trained_weights=pickle.load(f)
  trained_weight=torch.from_numpy(trained_weights).float().to(device)

  vocab=pickle.load(open('/home/pickle_objects/vocab_object_2.pkl','rb'))
  
  model=load_model(hidden_units=config['hidden_units'],device=device,weights=trained_weight)
  saved=torch.load('/home/saved_models/best_model_150_highest.pth')#map_location='cpu')
  model.load_state_dict(saved['model_state_dict'])

  c_train=pickle.load(open('/home/pickle_objects/pad_2/c_train_pad.pkl','rb'))
  q_train=pickle.load(open('/home/pickle_objects/pad_2/q_train_pad.pkl','rb'))
  ans_train=pickle.load(open('/home/pickle_objects/pad_2/ans_train_pad.pkl','rb'))  

  c_test=pickle.load(open('/home/pickle_objects/pad_2/c_test_pad.pkl','rb'))
  q_test=pickle.load(open('/home/pickle_objects/pad_2/q_test_pad.pkl','rb'))
  ans_test=pickle.load(open('/home/pickle_objects/pad_2/ans_test_pad.pkl','rb'))

  c_val=pickle.load(open('/home/pickle_objects/pad_2/c_val_pad.pkl','rb'))
  q_val=pickle.load(open('/home/pickle_objects/pad_2/q_val_pad.pkl','rb'))
  ans_val=pickle.load(open('/home/pickle_objects/pad_2/ans_val_pad.pkl','rb'))  

  ans_start,ans_end=get_answer_index(c_val,ans_val)

  dl_test=batched_data(c_test,q_test,ans_test,batch=8192,num_workers=7)
  dl_val=batched_data(c_val,q_val,ans_val,batch=8192,num_workers=7)
  dl_train=batched_data(c_train,q_train,ans_train,batch=8192,num_workers=7)
  dl_val_=batched_indexed_data(c_val,q_val,ans_val,ans_start,ans_end,batch=8192,num_workers=7)
  


  t1=perf_counter()
  # inspecting model predictions
  with torch.no_grad():
    attn_matrix=model(context=c_val.to(device),question=q_val.to(device)).to(device)
    print(attn_matrix[4])
    #print(ans_sta)

  t2=perf_counter()
  print('run time: ',t2-t1)  
  print(list(zip(ans_start[4],ans_end[4])))
  print(len(c_val[4][c_val[4]!=0]))
  print(tokens_to_sentence([c_val[4]],vocab))
  print(tokens_to_sentence([q_val[4]],vocab))
  print(tokens_to_sentence([ans_val[4]],vocab))
     
    


