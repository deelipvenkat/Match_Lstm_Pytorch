"""training loop with checkpointing (pause/resume training)"""
import torch
import torch.nn.functional as F
import sys
import os
from torch.cuda.amp import autocast , GradScaler
from tqdm import tqdm
from utils import Early_Stopping, SaveBestModel , save_graphs,last_saved_model,good_saved_model
from evaluation import validate,validation_loss , validate_f1, validate_loss
import yaml
from torch.utils.data import DataLoader
import random
import math
import pickle
from torch.utils.tensorboard import SummaryWriter

device=['cuda' if torch.cuda.is_available() is True else 'cpu'][0]

with open('/home/config.yaml') as f:
  config=yaml.safe_load(f)
#-------------------------------------------------------------------------------------------------
def my_collate(batch):
  a=list(zip(*batch))
  print(type(a[0]),type(a[1]),type(a[2]),type(a[3]))
  a0,a1 = torch.stack(a[0],0), torch.stack(a[1], 0) 
  return a0,a1,a[2],a[3]


def re_trainer(epochs,model,optimizer
            ,train_dataset,val_dataloader,
            train_loss_record=[],val_loss_record=[],
            f1_score_record=[],epoch_record=0,seed_value=0,batches_trained_epoch=0):
    
  scaler=GradScaler() # USING AVERAGE-MIXED-PRECISION
  
  best_loss_=min(val_loss_record) if len(val_loss_record)!=0 else float('inf')
  save_best_model = SaveBestModel(best_loss=best_loss_)
  
  if os.path.exists('/home/training_data/f1_high.pkl'):
    high_f1_record=pickle.load(open('/home/training_data/f1_high.pkl','rb'))
    print('pickle file found')

  else:
    high_f1_record=[]

  val_loss=float('inf')
  f1_best=0    


  early_stopping=Early_Stopping(patience=config['early_stopping_patience'],min_delta=config['early_stopping_delta'])
  
  writer=SummaryWriter('/home/runs/squad')
  total_iteration=0
  restart=True

  #  train-data-input , last index , shuffling , random state , dataloader prep(without shuffle)
  # dataloader contains context/question/ans start/end index.

  for epoch in range(epoch_record,epochs):
    
    random.seed(seed_value) # save seed
    indices=list(range(len(train_dataset)))
    random.shuffle(indices)
    
    batches_trained_epoch= 0 if restart is False else batches_trained_epoch
    batches_left_before_training=batches_trained_epoch   

    if (restart is True) and (config['checkpoint_exists'] is True):

      
      print('re-starting the model from previous saved state from randomized index {}'.format(int(config['train_batch_size'])*batches_trained_epoch))
      subset_data=torch.utils.data.Subset(train_dataset,indices[(batches_trained_epoch)*int(config['train_batch_size']):])
      
      train_dataloader=DataLoader(dataset=subset_data,batch_size=config['train_batch_size'],
                            shuffle=False,num_workers=config['num_workers']
                            ,pin_memory=True)

      restart=False
    
    else:
      restart=False
      
      subset_data=torch.utils.data.Subset(train_dataset,indices=indices)

      train_dataloader=DataLoader(dataset=subset_data,batch_size=config['train_batch_size'],
                            shuffle=False,num_workers=config['num_workers']
                            ,pin_memory=True)

    
    tq=tqdm(train_dataloader,colour='green')
    tq.set_description('epoch %s' %(epoch+1))
    running_loss_per_epoch=0
    
    for context,question,start,end in tq:
      
      model.train()
      current_batch_size=context.size(0)
      
      context,question,start,end=context.to(device),question.to(device),start.to(device),end.to(device)
      
      with autocast():
              
        attention_matrix=model(context=context,question=question)
        # attention_matrix=exp_mask(attention_matrix,c_train_l)

        y_pred=torch.empty(current_batch_size)     
        
        for i in range(current_batch_size):
          
          y_pred[i]=attention_matrix[i,start[i],0]*attention_matrix[i,end[i],1]

        loss=F.binary_cross_entropy(y_pred,torch.ones(current_batch_size)).to(device)
        
      optimizer.zero_grad()
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      loss_=loss.item()  
      train_loss_record.append(loss_)
      running_loss_per_epoch+= loss_*current_batch_size
      writer.add_scalar('train_loss/iteration',loss_,total_iteration)
      batches_trained_epoch+=1 

      if ((total_iteration+1)%config['model_checkpointing_iteration'])==0:
        
        with torch.no_grad():
          
          model.eval()
          val_loss=validate_loss(dl=val_dataloader,model=model)
          val_loss_record.append(val_loss)
          writer.add_scalar('validation_loss',val_loss,(total_iteration//config['model_checkpointing_iteration'])+1)

          if early_stopping is True :
            print('EARLY STOPPING ACTIVATED')  
            break

        if val_loss>3 :
           
          save_best_model(val_loss,epoch,model,optimizer,train_loss_record,
            val_loss_record,f1_score_record,seed_value,batches_trained_epoch) # checkpointing low val_loss model

        else :

          f1_best=validate_f1(dt=val_dataloader,models=model)
          score=max(high_f1_record) if len(high_f1_record)>0 else 0
          if f1_best>=score:
            high_f1_record.append(f1_best)
            
            pickle.dump(high_f1_record,open('/home/training_data/f1_high.pkl','wb'))
            
            good_saved_model(epoch,model,optimizer,train_loss_record,
              val_loss_record,f1_score_record,f1_best,seed_value,batches_trained_epoch)

        last_saved_model(epoch,model,optimizer,train_loss_record,
          val_loss_record,f1_score_record,seed_value,batches_trained_epoch)

          

      total_iteration+=1
          
      tq.set_postfix(train_loss=loss_,val_loss=val_loss,best_f1=f1_best) # tqdm printing of loss

    print('running loss for epoch {}: '.format(epoch+1), running_loss_per_epoch/(len(train_dataset)-(config['train_batch_size']*batches_left_before_training))) # loss per epoch avg
    
    seed_value+=1
    
    with torch.no_grad():
      f1=validate_f1(dt=val_dataloader,models=model)
      f1_score_record.append(f1)
      print(" validation f1_score for epoch {}".format(epoch+1),f1*100)
      writer.add_scalar('f1_score/epoch',f1,epoch+1)

  writer.close()
  
  torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, config['save_final_model_path']) # final sav  

  save_graphs(train_loss_record,val_loss_record,f1_score_record)

#-------------------------------------------------------------------------------------------------

if __name__=='__main__':
  pass    
