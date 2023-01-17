import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import torch.nn.functional as F
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy.lang.en import English
from spacy.tokens import DocBin
nlp=English()

with open('/home/config.yaml') as f:
  config=yaml.safe_load(f)

device=['cuda' if torch.cuda.is_available() is True else 'cpu'][0]

#-------------------------------------------------------------------------------------------------

"""
#df_train=pd.read_csv('/home/datasets/squad.csv')

lis=[]
for it,i in enumerate(range(df_train.shape[0])):
    start_no=df_train['answer_start'][it]
    res = list(df_train['context'][it])
    res.insert(start_no, '@@@')
    res = ''.join(res)
    lis.append(res)
df_train['edited_context']=lis
"""
def tokenizer_2(h):
  """
  tokenizer for train set - using spacy tokenizer
  """

  a=[[],[],[]]
  nlp = English()
  # Create a blank Tokenizer with just the English vocab
  x=['edited_context','question','answer_text']
  for k,i in enumerate(x):
    for _ in range(h.shape[0]):
      words = nlp(str(h[i][_]).lower())
      tokens = [word.text for word in words]
      a[k].append(tokens)
  
  return a[0],a[1],a[2]

#c_t,q_t,a_t=tokenizer_2(df_train)

def ans_start_real(p,q):
    p_,q_=[],[]
    for i_ in range(len(p)):
        for k,i in enumerate(p[i_]):
            if "@@@" in i:
                p_.append(k)
                q_.append(k+len(q[i_])-1)        
            else:
                pass
    return p_,q_
"""
final,end=ans_start_real(c_t,a_t)

for k,i in enumerate(final):
    if i>399:
        final[k]=398
    else:
        pass
        
for k,i in enumerate(end):
    if i>399:
        end[k]=398
    else:
        pass
"""
#st=pickle.load(open('/home/pickle_objects/start_index.pkl','rb'))
#en=pickle.load(open('/home/pickle_objects/end_index.pkl','rb'))    


#-------------------------------------------------------------------------------------------------
def spacytoken_to_string (context,start,end):
  
  """
  returns a list of answers used to slice exact answer form question, takes in spacy tokenized 
  text object , answer start/end index.
  """

  h=[]
  for i in range(len(context)):
    stri=''.join([token.text_with_ws for token in context[i][start[i]:end[i]+1]]).rstrip()
    h.append(stri)
  return h

#-------------------------------------------------------------------------------------------------
def answer_doc_list(p,answers_count):
  #docs = list(p.get_docs(nlp.vocab))
  c=0
  ans_words=[]
  for k,i in enumerate(answers_count):
      ans_words.append(p[c:c+i])
      c=i
  return ans_words
#-------------------------------------------------------------------------------------------------

# USED FOR FINDING ANSWER START AND END INDEX FROM CONTEXT AND ANSWERS-

def sublist(x,y):
  
  """
  helper function to compute answer start/end index in passage by performing match on each 
  sublist of passage.
  """

  l_=len(y)
  result=0,0
  for ind in (i for i,e in enumerate(x) if e==y[0]):

    if x[ind:ind+l_]==y:

      result = ind,ind+l_-1
  
  return result    

def answer_processor(c,a):

  """
  helper function to compute answer start/end index in passage
  """

  start_vector=[]
  end_vector=[]

  for i in range(c.size(0)):

    x,y=c[i].tolist(),a[i].tolist()

    c_=[i for i in x if i!=0] # 0 is the int value of <pad> in vocab - removes pad values
    a_=[i for i in y if i!=0]
    start_index,end_index=sublist(c_,a_)

    start_vector.append(start_index)
    end_vector.append(end_index)  

  return start_vector,end_vector  # shape- N same fir both start/end sets

#-------------------------------------------------------------------------------------------------

def get_answer_index(c_pad,ans_pad):

  """
  returns answer start & end index in passage as tuple of torch tensors, 
  taking padded context and answer torch tensors as inputs
  """

  holder=torch.zeros(c_pad.size(0),2)
  
  s,e=answer_processor(c_pad,ans_pad)
  a_start=torch.from_numpy(np.array(s)).unsqueeze(1)
  a_end=torch.from_numpy(np.array(e)).unsqueeze(1)
  
  #holder=torch.cat((a_start,a_end),dim=1)
  
  return a_start,a_end

#-------------------------------------------------------------------------------------------------


def seq_length(p):

    s_=[]
    for k,i in enumerate(p):
        s_.append(len(i[i!=0]))
    return s_

#-------------------------------------------------------------------------------------------------


def last_saved_model(epoch,model,optimizer,train_losses
                    ,val_losses,f1_scores,seed_value,batches_trained):
  if True:

    torch.save({'epoch': epoch, # FIX : EPOCH+1
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss_data':train_losses,
    'val_loss_data': val_losses,
    'f1_score_data':f1_scores,
    'seed_value':seed_value,
    'batches_trained':batches_trained},config['last_model_path'])


#-------------------------------------------------------------------------------------------------


def good_saved_model(epoch,model,optimizer,train_losses
                    ,val_losses,f1_scores,best_f1,seed_value,batches_trained):
  if True:

    torch.save({'epoch': epoch, # FIX : EPOCH+1
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss_data':train_losses,
    'val_loss_data': val_losses,
    'f1_score_data':f1_scores,
    'best_f1':best_f1,
    'seed_value':seed_value,
    'batches_trained':batches_trained},config['good_model_path'])

#-------------------------------------------------------------------------------------------------
# model checkpointing

class SaveBestModel:
    
    """
    Class to save the best model while training. If the current iteration's 
    validation loss is less than the previous lowest loss, then save the
    model state.
    """
    def __init__(self,best_loss=float('inf')):
        self.best_loss = best_loss
        
    def __call__(self, current_loss,epoch,model, optimizer,
            train_losses,val_losses,f1_scores,seed_value,batches_trained):
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            torch.save({'epoch': epoch, # FIX : EPOCH+1
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_data':train_losses,
            'val_loss_data': val_losses,
            'f1_score_data':f1_scores,
            'seed_value':seed_value,
            'batches_trained':batches_trained},config['save_best_model_path']) # use config
#-------------------------------------------------------------------------------------------------

def load_checkpoint(model,optimizer):

  checkpoint=torch.load(config['checkpoint_path'])
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  train_loss_=checkpoint['train_loss_data']
  val_loss_=checkpoint['val_loss_data']
  f1_loss_=checkpoint['f1_score_data']
  ep=checkpoint['epoch']
  seed_value=checkpoint['seed_value']
  batches_completed=checkpoint['batches_trained']


  return model,optimizer,train_loss_,val_loss_,f1_loss_,ep,seed_value,batches_completed


#-------------------------------------------------------------------------------------------------

class Early_Stopping:

  """
  Performs early stopping if validation loss is above a certain threshold(min delta) from least
  loss value for certain number of iterations(patience).
  
  """

  def __init__(self, patience, min_delta):
    self.patience=patience
    self.min_delta=min_delta
    self.best_loss= float('inf')
    self.counter=0
    
  def __call__(self,val_loss):
    if val_loss< self.best_loss:
      self.best_loss=val_loss
      self.counter=0

    elif val_loss> self.best_loss+self.min_delta:
      
      self.counter+=1
      
      if self.counter>self.patience:
        return True  
    
    return False

#-------------------------------------------------------------------------------------------------

def save_graphs(train_loss,val_loss,f1_score):
  plt.style.use('dark_background')
  plt.rcParams["figure.figsize"] = [10, 7]

  plt.figure(1)
  plt.plot(train_loss,label='training loss',color='red')
  plt.xlabel('iterations')
  plt.ylabel('train loss')
  plt.title('training loss')
  plt.savefig('/home/training_data/train_loss.png')
  
  plt.figure(2)
  plt.plot(val_loss,label='validation loss',color='blue')
  plt.xlabel('iterations')
  plt.ylabel('val loss')
  plt.title('validation loss')
  plt.savefig('/home/training_data/val_loss.png')
  
  plt.figure(3)
  plt.plot(f1_score,label='F1-scores')
  plt.xlabel('iterations')
  plt.ylabel('f1-score')
  plt.title('f1-scores/epoch')  
  plt.savefig('/home/training_data/f1_score.png')

#-------------------------------------------------------------------------------------------------
def exponential_mask(p,seq_le,partial=False):
    """
    takes in attention with logits , applies softmax over the sequence
    length & sets the probabilites of pad values to 0
    """
    vect=torch.zeros(p.size(0),400,2,device=device)
    for j in range(p.size(0)):
        l_=seq_le[j]

        vect[j][:l_,:]=p[j][:l_,:]

        if partial is True:
            for i in range(l_,400):
                vect[j][i,:]=torch.tensor([-2,-2],device=device)

            vect[j][:,0],vect[j][:,1]=F.softmax(vect[j][:,0],dim=0),F.softmax(vect[j][:,1],dim=0)

        else:
            vect[j][:l_,0],vect[j][:l_,1]=F.softmax(p[j][:l_,0],dim=0),F.softmax(p[j][:l_,1],dim=0)
  
    return vect
#-------------------------------------------------------------------------------------------------

if __name__=='__main__':
  print(sublist.__doc__) 
