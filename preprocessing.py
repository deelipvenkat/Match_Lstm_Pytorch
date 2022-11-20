import numpy as np
import pickle
import yaml
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
import torch 
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English, EnglishDefaults
from dataloader import batched_data

with open('/home/config.yaml') as f:
  config=yaml.safe_load(f)


#------------------------------------------------------------------------------------------------#

def json_to_df(j): # used for train set 

  data=pd.DataFrame(columns=['title','context','question','answer_start','answer_text'])
  
  o_=len(j['data'])

  for h in range(o_): # title
    l_=len(j['data'][h]['paragraphs'])
    title=j['data'][h]['title']
  
    for i in range(l_): # passage
      context=j['data'][h]['paragraphs'][i]['context']
      k_=len(j['data'][h]['paragraphs'][i]['qas'])
  
      for j_ in range(k_): # question
        que=j['data'][h]['paragraphs'][i]['qas'][j_]['question']
        ans_s=j['data'][h]['paragraphs'][i]['qas'][j_]['answers'][0]['answer_start']
        ans_t=j['data'][h]['paragraphs'][i]['qas'][j_]['answers'][0]['text']
  
        data.loc[len(data.index)]=[title,context,que,ans_s,ans_t]
  
  return data

#------------------------------------------------------------------------------------------------#

def dev_json_to_df(j): # used for test set with multiple answers.

  data=pd.DataFrame(columns=['title','context','question','answer_text'])
  
  o_=len(j['data'])

  for h in range(o_): # title
    l_=len(j['data'][h]['paragraphs'])
    title=j['data'][h]['title']
  
    for i in range(l_): # passage
      context=j['data'][h]['paragraphs'][i]['context']
      k_=len(j['data'][h]['paragraphs'][i]['qas'])
  
      for j_ in range(k_): # question
        que=j['data'][h]['paragraphs'][i]['qas'][j_]['question']
      
        ans_l=len(j['data'][h]['paragraphs'][i]['qas'][j_]['answers'])
        multiple=[]
        
        for m_ in range(ans_l): # loop for multiple answers
          ans_t=j['data'][h]['paragraphs'][i]['qas'][j_]['answers'][m_]['text']
          multiple.append(ans_t)
        
        unrepeat=list(set(multiple)) # handles repetitive answers by excluding them.

        data.loc[len(data.index)]=[title,context,que,unrepeat]
    
    return data            

#------------------------------------------------------------------------------------------------#

def splitter(df,val_ratio):
  
  """
  this is used to split dataframe to desired ratio,
  shuffle is currently set to False
  """

  q_train,q_val,c_train,c_val,a_train,a_val=train_test_split(df['question'],df['context'],df['answer_text'],test_size=val_ratio,random_state=0,shuffle=False)
  
  d1=pd.DataFrame(list(zip(c_train,q_train,a_train)),columns=['context','question','answer_text'])
  d2=pd.DataFrame(list(zip(c_val,q_val,a_val)),columns=['context','question','answer_text'])

  return d1,d2

#------------------------------------------------------------------------------------------------#
# spacy tokenizer is good, but splitting at aprostrophe/contraction words

def tokenizer(h):
  """
  tokenizer for train set - using spacy tokenizer
  """

  a=[[],[],[]]
  nlp = English()
  # Create a blank Tokenizer with just the English vocab
  x=['context','question','answer_text']
  for k,i in enumerate(x):
    for _ in range(h.shape[0]):
      words = nlp(str(h[i][_]))# used to be lower
      tokens = [word.text.lower() for word in words] # text to convert spacy tokens to words
      a[k].append(tokens)
  
  return a[0],a[1],a[2]

#------------------------------------------------------------------------------------------------#

# pipeline converting string to integers for context
def word2_int(h,vocab):
  
  """
  converts word to integer using vocabulary dictionary index
  """

  text_pipeline = lambda x: vocab(x)
  q=[]
  for i in h:
    q.append(text_pipeline(i))
  return q  


#------------------------------------------------------------------------------------------------#
# pads the embeddings and truncates to a fixed value.

class pad_truncate:

  """
  paddding and truncating of context , question , answer to a fixed values set.(400/30/20)
  """

  def __init__(self,seq_length):
    self.seq_length=seq_length
  
  def pad_features(self,length_int):
    features = np.zeros((len(length_int), self.seq_length), dtype = int)
    
    for i, text in enumerate(length_int):
      text_len = len(text)
        
      if text_len <= self.seq_length:
            zeroes = list(np.full((self.seq_length-text_len),0))
            new = text+zeroes
      elif text_len > self.seq_length:
            new = text[0:self.seq_length]
           
      features[i,:] = np.array(new)
    
    return features

#------------------------------------------------------------------------------------------------#
# takes tokenized text as input
class pad_pipeline:

  """
  implements word to integer , padding/truncating , numpy to torch tensor conversion
  
  """
  
  def __init__(self,pad_feature_context,pad_feature_question,pad_feature_answer,vocab):
    self.vocab=vocab
    self.pad_feature_context=pad_feature_context
    self.pad_feature_question=pad_feature_question
    self.pad_feature_answer=pad_feature_answer

  def question_processing(self,x):
    
    x=word2_int(x,self.vocab)
    x=self.pad_feature_question.pad_features(x)
    x=torch.from_numpy(x).long()
    
    return x

  def context_processing(self,x):
    
    x=word2_int(x,self.vocab)
    x=self.pad_feature_context.pad_features(x)
    x=torch.from_numpy(x).long()
    
    return x    

  def answer_processing(self,x):
    
    x=word2_int(x,self.vocab)
    x=self.pad_feature_answer.pad_features(x)
    x=torch.from_numpy(x).long()
    
    return x 

#------------------------------------------------------------------------------------------------#
# split the dataframe if neccesary and get the padded data

def train_preprocessor(df,vocab):

  """
  takes in dataframe and returns a padded text of context/question/answer of TRAIN SET
  tokenization/padding/truncation/word2_int/conversion to torch tensor.
  """

  c_train,q_train,ans_train=tokenizer(df)
  
  pad_feature_context=pad_truncate(400) # context/question/answer length
  pad_feature_question=pad_truncate(30)
  pad_feature_answer=pad_truncate(20)

  pipe=pad_pipeline(pad_feature_context,pad_feature_question,pad_feature_answer,vocab)

  c_pad,q_pad,ans_pad=pipe.context_processing(c_train),pipe.question_processing(q_train),pipe.answer_processing(ans_train)
  
  return c_pad,q_pad,ans_pad

#------------------------------------------------------------------------------------------------#

def answer_tokenizer(h): # takes in
  
  """
  used for tokenization of test set answers. takes in dataframe series after processed using
  df.apply(eval)
  """

  h=list(h)
  a=[]
  nlp = English()
  # Create a blank Tokenizer with just the English vocab
  
  for i in range(len(h)):
    c_=[]
    for j in range(len(h[i])):
      words = nlp(str(h[i][j]))
      tokens = [word.text.lower() for word in words]
      c_.append(tokens)
    a.append(c_)  
  
  return a

#------------------------------------------------------------------------------------------------#  

def val_set_tokenizer(h):

  """
  test set tokenization of context/question , takes in a dataframe containing context/question
  and returns tokenized text as list
  """

  a=[[],[]]
  nlp = English()
  # Create a blank Tokenizer with just the English vocab
  x=['context','question']
  for k,i in enumerate(x):
    for _ in range(h.shape[0]):
      words = nlp(str(h[i][_]))
      tokens = [word.text.lower() for word in words]
      a[k].append(tokens)
  
  return a[0],a[1]

#------------------------------------------------------------------------------------------------#  

def test_preprocessor(df,vocab): # processor only question and answer
  # maximum length in dev set is 5 ,so we pad answer based on that.

  """
  takes in dataframe and returns a padded text of context/question of TEST SET.
  tokenization/padding/truncation/word2_int/conversion to torch tensor.
  """  

  c_val,q_val=val_set_tokenizer(df)
  
  pad_feature_context=pad_truncate(400) # context/question/answer length
  pad_feature_question=pad_truncate(30)
  pad_feature_answer=pad_truncate(20)

  pipe=pad_pipeline(pad_feature_context,pad_feature_question,pad_feature_answer,vocab)

  c_pad,q_pad=pipe.context_processing(c_val),pipe.question_processing(q_val)

  return c_pad,q_pad

#------------------------------------------------------------------------------------------------#

# for final processing of answers in test set
def test_answer_processing(h,vocab): # h is list of set of answers

  """
  takes in LIST of answers of test set and outputs a torch tensor of padded -processed answers
  which are of shape BATCH*5*20.
  """
  
  text_pipeline = lambda x: vocab(x) # using [] since this accepts [] versions as input

  a=torch.empty((len(h),5,20))
  for i in range(len(h)):
    c_=[]
    x=[]
    for j in range(len(h[i])):
      pad_feature_answer=pad_truncate(20)
      
      per_ans=[]
      #per_ans.append(text_pipeline(k)[0])
      x.append(text_pipeline(h[i][j]))


    x_=pad_feature_answer.pad_features(x)

    #print(x_)
    
    if len(h[i])<5:
      x_=np.array(x_)
      c_=torch.from_numpy(x_)
      #print(c_)
      #sys.exit()
      miss=5-len(h[i])
      #print(miss,len(h[i]))
      v_=torch.full((miss,20),0)
      #print(v_)
      ans_set=torch.vstack([c_,v_])
      #print(ans_set)
      

    else:
      x_=np.array(x_)
      c_=torch.from_numpy(x_)      
      ans_set=c_
      pass
      
    a[i,:,:]=ans_set

  return a

#------------------------------------------------------------------------------------------------#

if __name__=='__main__':

  df=pd.read_csv('/home/datasets/squad.csv')
  #df_test=pd.read_csv('/home/datasets/squad_test.csv')
  
  #df_test['answer_text']=df_test['answer_text'].apply(eval)
  #vocab=pickle.load(open('pickle_objects/vocab_object.pkl','rb'))
  
  df_train,df_val=splitter(df,0.05)


  #c_train_pad,q_train_pad,ans_train_pad=train_preprocessor(df_val,vocab)
  #c_val,q_val=test_preprocessor(df_test,vocab)
  
  #c_=answer_tokenizer(df_test['answer_text']) # tokenization

  #ans_val=test_answer_processing(c_,vocab)

  #dt=batched_data(c_val,q_val,ans_val,batch=10,num_workers=1)




    