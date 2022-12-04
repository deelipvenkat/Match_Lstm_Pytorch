import torch
from models import load_model
import pickle
from spacy.lang.en import English

device='cpu'
nlp=English()


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



def proba_score_search(p):

  
  holder=torch.empty(p.size(0),2,device='cpu')
  for i in range(p.size(0)):

    start=torch.argmax(p[i,:,0])
    end=torch.argmax(p[i,:,1])
    
    if end-start<=15 and end>=start:
      pass
    
    elif start>end or end-start>15:
      start,end=search_method(p[i])         

    holder[i,:]=torch.tensor([start,end])

  return holder

def word2_int(h,vocab):
  
  """
  converts word to integer using vocabulary dictionary index
  """

  text_pipeline = lambda x: vocab(x)
  q=[]
  for i in h:
    q.append(text_pipeline([i])[0])
  return q


def input_processing(s1,s2,vocab):
    s1=nlp(s1)
    s2=nlp(s2)

    context_tokens = [word.text.lower() for word in s1]
    question_tokens = [word.text.lower() for word in s2]
    context=torch.tensor(word2_int(context_tokens,vocab),dtype=torch.int32)
    question=torch.tensor(word2_int(question_tokens,vocab),dtype=torch.int32)
    c_pad=torch.hstack((context,torch.zeros(400,dtype=torch.int32)))[:400]
    q_pad=torch.hstack((question,torch.zeros(30,dtype=torch.int32)))[:30]

    return c_pad,q_pad,s1

