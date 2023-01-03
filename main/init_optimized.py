"""top level module to start training the network with checkpointing 
   & optimized with bucket-batch-sampler"""
import pandas as pd
import numpy as np
import pickle
import yaml
import sys

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from preprocessing import splitter,train_preprocessor
from models.models_optimized import load_model
from utils import get_answer_index,seq_length,load_checkpoint
from dataloader import retrain_dataset_optimized,train_dataset,optimized_dataloader,optimized_val_dataloader
from train import trainer
from evaluation import validate,proba_score_search
from metrics import f1_score,em_score
from tqdm.auto import tqdm
from spacy.tokens import DocBin
from spacy.lang.en import English
from train.retrainer_optimized import re_trainer

#from models_bi_ans_ptr import load_model

if __name__=='__main__':

  with open('config.yaml') as f:
    config=yaml.safe_load(f)

  device=['cuda' if torch.cuda.is_available() is True else 'cpu'][0]

  f=open('/home/pickle_objects/fixed_objects/trained_weights_100.pkl','rb')
  trained_weights=pickle.load(f)
  trained_weight=trained_weights.to(device)

  vocab=pickle.load(open('/home/pickle_objects/fixed_objects/vocabulary.pkl','rb'))
    
  c_train_pad=pickle.load(open('/home/pickle_objects/fixed_objects/pad_objects/c_train_pad.pkl','rb'))
  q_train_pad=pickle.load(open('/home/pickle_objects/fixed_objects/pad_objects/q_train_pad.pkl','rb'))
  ans_train_pad=pickle.load(open('/home/pickle_objects/fixed_objects/pad_objects/ans_train_pad.pkl','rb'))

  c_val_pad=pickle.load(open('/home/pickle_objects/fixed_objects/pad_objects/c_val_pad.pkl','rb'))
  q_val_pad=pickle.load(open('/home/pickle_objects/fixed_objects/pad_objects/q_val_pad.pkl','rb'))
  ans_val_pad=pickle.load(open('/home/pickle_objects/fixed_objects/pad_objects/ans_val_pad.pkl','rb'))   

  match_model=load_model(hidden_units=config['hidden_units'],device=device,weights=trained_weight)

  pytorch_all_params = sum(p.numel() for p in match_model.parameters())
  pytorch_total_params = sum(p.numel() for p in match_model.parameters() if p.requires_grad)

  print('total parameters for match lstm: ',pytorch_all_params)
  print('trainable parameters for match lstm :',pytorch_total_params)

  optimizer=torch.optim.Adam(match_model.parameters(),lr=config['learning_rate'])

  c_train_len,q_train_len=seq_length(c_train_pad),seq_length(q_train_pad) # inputs spacy tokens
  c_val_len,q_val_len=seq_length(c_val_pad),seq_length(q_val_pad)

  start=pickle.load(open('/home/pickle_objects/fixed_objects/start_index.pkl','rb'))
  end=pickle.load(open('/home/pickle_objects/fixed_objects/end_index.pkl','rb'))

  ans_start,ans_end=start[:83219],end[:83219]      
  ans_val_start,ans_val_end=start[83219:],end[83219:]
 
  batch_train_data=optimized_dataloader(c_train_pad,q_train_pad,
          ans_start,ans_end,c_train_len,batch_size=config['train_batch_size'])      

  batch_val_data=optimized_val_dataloader(context=c_val_pad,question=q_val_pad,
          answer=ans_val_pad,start=ans_val_start,end=ans_val_end,length=c_val_len,batch_size=config['val_batch_size'])  

  unpad_context=[]
  for k,i in enumerate(c_train_pad):
    v=i[i!=0]
    unpad_context.append(v)

  training_dataset=retrain_dataset_optimized(unpad_context, q_train_pad, ans_start[:83219], ans_end[:83219],c_train_len)

      

  # write a load checkpoint function which looks cleaner here.    
  if config['checkpoint_exists']:
        
        print('saved checkpoint found')

        (match_model,optimizer,train_loss_
        ,val_loss_,f1_loss_,ep,seed_value,batches_trained)=load_checkpoint(match_model,optimizer)

        re_trainer(epochs=config['epochs'],model=match_model,optimizer=optimizer,
                train_dataset=training_dataset,
                val_dataloader=batch_val_data,train_loss_record=train_loss_,
                val_loss_record=val_loss_,f1_score_record=f1_loss_,
                epoch_record=ep,seed_value=seed_value,batches_trained_epoch=batches_trained)        
      
  else:

        data_zip=list(zip(unpad_context,q_train_pad,ans_start[:83219],ans_end[:83219],c_train_len))
        sorted_data = sorted(data_zip, key=lambda x: x[4], reverse=False)
        unpad_context_s,q_train_s,start_index_s,end_index_s,c_train_len_s=zip(*sorted_data)

        fast_dataloader=optimized_dataloader(context=unpad_context_s,question=q_train_s,start=start_index_s,
        end=end_index_s,length=c_train_len_s,batch_size=config['train_batch_size'])
        
        
        print('no saved checkpoint found')
        re_trainer(epochs=config['epochs'],model=match_model,optimizer=optimizer,
                train_dataset=training_dataset,
                val_dataloader=batch_val_data,optimized_data=fast_dataloader)   