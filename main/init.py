"""top level module to start training the network"""

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
from dataloader import batched_data,batched_indexed_data
from train.train import trainer
from evaluation import validate,proba_score_search
from metrics import f1_score,em_score
from tqdm.auto import tqdm
from spacy.tokens import DocBin
from spacy.lang.en import English

#from models_bi_ans_ptr import load_model

if __name__=='__main__':

  with open('config.yaml') as f:
    config=yaml.safe_load(f)

  device=['cuda' if torch.cuda.is_available() is True else 'cpu'][0]

  f=open(config['trained_weights_path'],'rb')
  trained_weights=pickle.load(f)
  trained_weight=torch.from_numpy(trained_weights).float().to(device)

  vocab=pickle.load(open(config['vocab_path'],'rb'))
    
  c_train_pad=pickle.load(open('/home/pickle_objects/pad_2/c_train_pad.pkl','rb'))
  q_train_pad=pickle.load(open('/home/pickle_objects/pad_2/q_train_pad.pkl','rb'))
  ans_train_pad=pickle.load(open('/home/pickle_objects/pad_2/ans_train_pad.pkl','rb'))

  c_val_pad=pickle.load(open('/home/pickle_objects/pad_2/c_val_pad.pkl','rb'))
  q_val_pad=pickle.load(open('/home/pickle_objects/pad_2/q_val_pad.pkl','rb'))
  ans_val_pad=pickle.load(open('/home/pickle_objects/pad_2/ans_val_pad.pkl','rb'))

  c1=DocBin().from_disk('/home/pickle_objects/fixed_objects/tok_context_train')
  c2=DocBin().from_disk('/home/pickle_objects/fixed_objects/tok_context_test')

  nlp=English()

  c1_=list(c1.get_docs(nlp.vocab))
  #c2_=list(c2.get_docs(nlp.vocab))    
  spacy_context_train=c1_[:83219]
  spacy_context_val=c1_[:83219]      

  match_model=load_model(hidden_units=config['hidden_units'],device=device,weights=trained_weight)

  pytorch_all_params = sum(p.numel() for p in match_model.parameters())
  pytorch_total_params = sum(p.numel() for p in match_model.parameters() if p.requires_grad)

  print('total parameters for match lstm: ',pytorch_all_params)
  print('trainable parameters for match lstm :',pytorch_total_params)

  optimizer = torch.optim.Adam(match_model.parameters(), lr=config['learning_rate'])
  #optimizer=torch.optim.Adamax(match_model.parameters(),lr=config['learning_rate'])

  c_train_len,q_train_len=seq_length(c_train_pad),seq_length(q_train_pad) # inputs spacy tokens
  c_val_len,q_val_len=seq_length(c_val_pad),seq_length(q_val_pad)
  #c_test_len,q_test_len=seq_length(c_test),seq_length(q_test) 

  start=pickle.load(open('/home/pickle_objects/start_index.pkl','rb'))
  end=pickle.load(open('/home/pickle_objects/end_index.pkl','rb'))

  ans_start,ans_end=start[:83219],end[:83219]      
  ans_val_start,ans_val_end=start[83219:],end[83219:]

  #docs_1 = list(doc_c_train.get_docs(nlp.vocab))

 
  batch_train_data=batched_indexed_data(c_train_pad,q_train_pad,
          ans_train_pad,ans_start,ans_end,batch=config['train_batch_size'],num_workers=config['num_workers'])

  batch_val_data=batched_indexed_data(c_val_pad,q_val_pad,
          ans_val_pad,ans_val_start,ans_val_end,batch=config['val_batch_size'],num_workers=config['num_workers'])

  # write a load checkpoint function which looks cleaner here.    
  if config['checkpoint_exists']:
        
        print('saved checkpoint found')

        (match_model,optimizer,train_loss_
        ,val_loss_,f1_loss_,ep)=load_checkpoint(match_model,optimizer)

        trainer(epochs=config['epochs'],model=match_model,optimizer=optimizer,
                train_dataloader=batch_train_data,
                val_dataloader=batch_val_data,train_loss_record=train_loss_,
                val_loss_record=val_loss_,f1_score_record=f1_loss_,epoch_record=ep)        
      

  else:
        print('no saved checkpoint found')
        trainer(epochs=config['epochs'],model=match_model,optimizer=optimizer,
                train_dataloader=batch_train_data,
                val_dataloader=batch_val_data)
        


   