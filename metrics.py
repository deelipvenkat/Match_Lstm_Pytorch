"""
Module used to compute F1/EM SCORE 

FUNCTIONS:

  f1_score(prediction,target)
  em_score(prediction,target)
"""

import sys
import yaml
import torch

with open('/home/config.yaml') as f:
  config=yaml.safe_load(f)
#-------------------------------------------------------------------------------------------------
def f1_score(pred,tar):
  
  if tar.ndim==2:
      
    s_=[]
    for i in range(len(pred)):
      prediction=pred[i]
      target=tar[i][tar[i]!=0]

      sa,sb=torch.unique(prediction),torch.unique(target)
      common_words=len([1 for i in sa if i in sb ])

      precision= common_words/len(prediction)
      recall= common_words/len(target)

      try :
        f1=(2*precision*recall)/(precision+recall)

      except ZeroDivisionError as err:
        f1=0  
      s_.append(f1)

    return sum(s_)/len(s_)

  elif tar.ndim==3:

    s_=[]
    for i in range(len(pred)):
      prediction=pred[i]
      scores_list=[]


      for t_ in tar[i]:
        
        target=t_[t_!=0]
        
        if len(target)==0:
          scores_list.append(0)
        
        else:

          sa,sb=torch.unique(prediction),torch.unique(target)
          common_words=len([1 for i in sa if i in sb ])
          
          precision= common_words/len(prediction)
          recall= common_words/len(target)

          try :
            f1=(2*precision*recall)/(precision+recall)
            scores_list.append(f1)

          except ZeroDivisionError as err:
            scores_list.append(0)

      f1_max=max(scores_list)
      s_.append(f1_max)

    return sum(s_)/len(s_)

  else:
    print('error')    


#*********************************************************************************

def em_score(pred,tar):
  
  
  if tar.ndim==2:  

    s_=0
    for i in range(len(pred)):

      target=tar[i][tar[i]!=0]
  
      if torch.equal(target,pred[i]) == True :
        s_+=1
      else :
        pass

    return s_/len(pred)

  elif tar.ndim==3:
    s_=0
    for i in range(len(pred)):

      scores_list=[]  

      for t_ in tar[i]:
        
        target=t_[t_!=0]

        if len(target)==0:
          scores_list.append(0)

        else:

          if torch.equal(target,pred[i]) == True:
            scores_list.append(1)
          else :
            scores_list.append(0)
            
      em_max=max(scores_list)
      s_=s_+ em_max        

    return s_/len(pred)


if __name__=='__main__':
  pass
