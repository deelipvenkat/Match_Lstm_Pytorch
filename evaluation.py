"""functions used to compute output probabilites of the model and validation loss"""

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
from dataloader import batched_data,batched_indexed_data
#from train import trainer
from metrics import f1_score , em_score

with open('/home/config.yaml') as f:
  config=yaml.safe_load(f)

device=['cuda' if torch.cuda.is_available() is True else 'cpu'][0]

#------------------------------------------------------------------------------------------------#

# creating a matrix of all possible start/end index probabilities.

def search_method(p):
    matrix=torch.zeros((400,15),device=device)
    for k,i in enumerate(p[:,0]):
        if i>0.001:
            a=i.unsqueeze(0)
            b = p[:,1][k:k+15].unsqueeze(0).to(device)
            c=torch.matmul(a,b).squeeze(0).to(device)
            pad=torch.zeros(15,device=device)
            if c.ndim==0:
                matrix[k,:]=pad
            else:

                c_=torch.cat((c,pad))[:15]

                matrix[k,:]=c_
            
        else:
            pass

    maximum=matrix.argmax()
    s=torch.div(maximum,matrix.size(1),rounding_mode='floor').item()
    e=(s+(maximum%matrix.size(1))).item()
        
    return s,e    
#------------------------------------------------------------------------------------------------#

def proba_score_search(p):

  
  holder=torch.empty(p.size(0),2,device=device)
  for i in range(p.size(0)):

    start=torch.argmax(p[i,:,0])
    end=torch.argmax(p[i,:,1])
    
    if end-start<=15 and end>=start:
      pass
    
    elif start>end or end-start>15:
      start,end=search_method(p[i])         

    holder[i,:]=torch.tensor([start,end])

  return holder  

#------------------------------------------------------------------------------------------------#

# greedy search for finding start and end index
def proba_score(p):
  # p is of the shape = batch*context_length*2
  """
  Used to compute max probability start/end index from attention matrix- output of the model
  """

  holder=torch.empty(p.size(0),2,device=device)
  for i in range(p.size(0)):
    start=torch.argmax(p[i,:,0])
    end=torch.argmax(p[i,:,1])
    
    if end<start:
      end=start
    else:
      pass

    holder[i,:]=torch.tensor([start,end])

  return holder # returns start and end index of each batch

#------------------------------------------------------------------------------------------------#


# returns sliced list of answer int values from context
def answer_from_context(c,answer_indexes):

  """
  returns sliced list of answer int values from context using output of 
  proba_score which are start/end index of answer. Mainly used for model inspection
  """

  l=[]
  if c.ndim==1 : c= c.unsqueeze(0)
  
  
  for i in range(c.size(0)):
 
    i_s,i_e=answer_indexes[i,:][0],answer_indexes[i,:][1]

    answer_=c[i][int(i_s):int(i_e)+1]
    l.append(answer_)
  return l
#------------------------------------------------------------------------------------------------#

# returns sentence string from vocab int list

def tokens_to_sentence(token,vocab): # accepts single tokens sentence
  """
  Used for converting tokens list to string sentence.Used for model inspection
  """
  l=" ".join(vocab.lookup_tokens(list(token[token!=0])))    
  return l
#------------------------------------------------------------------------------------------------#

def validate_f1(dt,models):
  length_1=[]

  for context_,question_,answer_,_,_ in dt:
    
    context_=context_.to(device)
    question_=question_.to(device)
    answer_=answer_.to(device)
    models.eval()
    attn_matrix=models(context=context_,question=question_).to(device)
    score=proba_score_search(attn_matrix)

    ans_list=answer_from_context(context_,score)
    
    f1_=f1_score(ans_list,answer_)
    length_1.append(f1_*context_.size(0))
      

  f1_total=sum(length_1)/len(dt.dataset)

  return f1_total


def validate(dt,models):

  """
  used to evaluate model on train/val/test sets , return tuple of f1_score and em_score.  
  """

  length_1,length_2=[],[]

  for context_,question_,answer_ in dt:
    
    context_=context_.to(device)
    question_=question_.to(device)
    answer_=answer_.to(device)
    models.eval()
    attn_matrix=models(context=context_,question=question_)
    score=proba_score_search(attn_matrix.to(device))
    ans_list=answer_from_context(context_,score)

    f1_=f1_score(ans_list,answer_)  
    em_=em_score(ans_list,answer_)

    length_1.append(f1_*context_.size(0))
    length_2.append(em_*context_.size(0))
    

  f1_total=sum(length_1)/len(dt.dataset)
  em_total=sum(length_2)/len(dt.dataset)  

  return f1_total,em_total

#------------------------------------------------------------------------------------------------#

def validation_loss(dt_val,model):
  
  """
  computes loss - takes in ans start/end index additionaly as inputs.
  """

  running_loss=0
  criterion=torch.nn.BCELoss()

  for context_,question_,_,ans_start_,ans_end_ in dt_val:
    
    batch_at_iteration=context_.size(0)

    if isinstance(ans_start_,list):

      context_,question_=context_.to(device),question_.to(device)

    else:
      context_,question_,ans_start_,ans_end_=context_.to(device),question_.to(device),ans_start_.to(device),ans_end_.to(device)
    
    y_pred=torch.empty(batch_at_iteration,device=device)
    model.eval()

    attn_matrix=model(context=context_,question=question_).to(device)

    for i in range(batch_at_iteration):
      
      y_pred[i]=attn_matrix[i,ans_start_[i],0]*attn_matrix[i,ans_end_[i],1]

    loss=criterion(y_pred,torch.ones(batch_at_iteration).to(device)).to(device)

    loss_=loss.item()
 
    running_loss+=loss_*batch_at_iteration
    
  val_loss=running_loss/len(dt_val.dataset)
  return val_loss

#------------------------------------------------------------------------------------------------#

def question_answer(model,context,question):
  """
  accepts list of context/question as input and return answer in string format
  """
  # preprocessing of context , question to torch-tensors
  #attn_matrix= model(context_,question_)
  #answers_index=proba_score(attn_matrix)
  
  # extract answers string from tokenized context by slicing.
  # create a loop for handling all the answers and add them to a list
  # function which can slice the exact answer string from given context using the sliced part from
  # tokenized context- need not be perfect , but decent
  
  #TEST YOUR BUILD WITH YOUR OWN PASSAGE AND QUESTION *********************************
#------------------------------------------------------------------------------------------------#


if __name__=='__main__':

  gc.collect()

  with open('config.yaml') as f:
    config=yaml.safe_load(f)

  device=['cuda' if torch.cuda.is_available() is True else 'cpu'][0]

  #f=open('/home/pickle_objects/squad_weight_2.pkl','rb')
  #trained_weights=pickle.load(f)
  #trained_weight=torch.from_numpy(trained_weights).float().to(device)
  
  f=open('/home/pickle_objects/fixed_objects/trained_weights_100.pkl','rb')
  trained_weights=pickle.load(f)
  trained_weight=trained_weights.to(device)
  
  vocab=pickle.load(open('/home/pickle_objects/fixed_objects/vocabulary.pkl','rb'))
  
  model=load_model(hidden_units=config['hidden_units'],device=device,weights=trained_weight)
  saved=torch.load('/home/best_model_150.pth')#map_location='cpu')
  model.load_state_dict(saved['model_state_dict'])

  c_train=pickle.load(open('/home/pickle_objects/fixed_objects/pad_objects/c_train_pad.pkl','rb'))
  q_train=pickle.load(open('/home/pickle_objects/pad_2/q_train_pad.pkl','rb'))
  ans_train=pickle.load(open('/home/pickle_objects/pad_2/ans_train_pad.pkl','rb'))  

  c_test=pickle.load(open('/home/pickle_objects/pad_2/c_test_pad.pkl','rb'))
  q_test=pickle.load(open('/home/pickle_objects/pad_2/q_test_pad.pkl','rb'))
  ans_test=pickle.load(open('/home/pickle_objects/pad_2/ans_test_pad.pkl','rb'))

  c_val=pickle.load(open('/home/pickle_objects/pad_2/c_val_pad.pkl','rb'))
  q_val=pickle.load(open('/home/pickle_objects/pad_2/q_val_pad.pkl','rb'))
  ans_val=pickle.load(open('/home/pickle_objects/pad_2/ans_val_pad.pkl','rb')) 

  #ans_start,ans_end=get_answer_index(c_test,ans_test)

  dl=batched_data(c_test,q_test,ans_test,batch=8192,num_workers=7)
  dl_val=batched_data(c_val,q_val,ans_val,batch=4096,num_workers=7)
  dl_train=batched_data(c_train,q_train,ans_train,batch=4096,num_workers=7)
  #dl_val_=batched_indexed_data(c_val,q_val,ans_val,ans_start,ans_end,batch=8192,num_workers=7)

  dl_test_=batched_data(c_test,q_test,ans_test,batch=4096,num_workers=7)
    
  t1=perf_counter()

  with torch.no_grad():
    f1,em=validate(dt=dl_test_, models=model.to(device))
  
  print('F1-SCORE: ',f1)
  print('EM-SCORE: ',em)
  
  t2=perf_counter()
  print(t2-t1)  
