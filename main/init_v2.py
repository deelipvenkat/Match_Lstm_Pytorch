"""top level module to start training the network with checkpointing"""

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
from models.models import load_model
from utils import get_answer_index,seq_length,load_checkpoint
from dataloader import batched_data,batched_indexed_data,retrain_dataset
from evaluation import validate,proba_score_search
from metrics import f1_score,em_score
from tqdm.auto import tqdm
from spacy.tokens import DocBin
from spacy.lang.en import English
from train.retrainer import re_trainer

#from models_bi_ans_ptr import load_model

if __name__=='__main__':

  with open('config.yaml') as f:
    config=yaml.safe_load(f)

  device=['cuda' if torch.cuda.is_available() is True else 'cpu'][0]

  f=open('/home/pickle_files/trained_embedding.pkl','rb')
  trained_weights=pickle.load(f)
  trained_weight=trained_weights.to(device)

  vocab=pickle.load(open('/home/pickle_files/vocab_object.pkl','rb'))

  c_train_pad=pickle.load(open('/home/pickle_files/pad/c_train_pad.pkl','rb'))
  q_train_pad=pickle.load(open('/home/pickle_files/pad/q_train_pad.pkl','rb'))
  ans_train_pad=pickle.load(open('/home/pickle_files/pad/ans_train_pad.pkl','rb'))

  c_val_pad=pickle.load(open('/home/pickle_files/pad/c_val_pad.pkl','rb'))
  q_val_pad=pickle.load(open('/home/pickle_files/pad/q_val_pad.pkl','rb'))
  ans_val_pad=pickle.load(open('/home/pickle_files/pad/ans_val_pad.pkl','rb'))

  start=pickle.load(open('/home/pickle_files/start_index.pkl','rb'))
  end=pickle.load(open('/home/pickle_files/end_index.pkl','rb'))

  match_model=load_model(hidden_units=config['hidden_units'],device=device,weights=trained_weight)

  pytorch_all_params = sum(p.numel() for p in match_model.parameters())
  pytorch_total_params = sum(p.numel() for p in match_model.parameters() if p.requires_grad)

  print('total parameters for match lstm: ',pytorch_all_params)
  print('trainable parameters for match lstm :',pytorch_total_params)

  optimizer=torch.optim.Adam(match_model.parameters(),lr=config['learning_rate'])

  training_dataset=retrain_dataset(c_train_pad,q_train_pad,start,end)

  #batch_val_data=batched_indexed_data(c_val_pad,q_val_pad,
    #      ans_val_pad,ans_val_start,ans_val_end
     #     ,batch=config['val_batch_size'],num_workers=config['num_workers'])
          
   
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
        print('no saved checkpoint found')
        re_trainer(epochs=config['epochs'],model=match_model,optimizer=optimizer,
                train_dataset=training_dataset,
                val_dataloader=batch_val_data)   