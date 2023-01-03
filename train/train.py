""" vanilla training loop"""
import torch
import torch.nn.functional as F
import sys
from torch.cuda.amp import autocast , GradScaler
from tqdm import tqdm
from utils import Early_Stopping, SaveBestModel , save_graphs,exponential_mask
from evaluation import validate,validation_loss , validate_f1
import yaml
from torch.utils.tensorboard import SummaryWriter

device=['cuda' if torch.cuda.is_available() is True else 'cpu'][0]

with open('/home/config.yaml') as f:
  config=yaml.safe_load(f)
#-------------------------------------------------------------------------------------------------

def trainer(epochs,model,optimizer
            ,train_dataloader,val_dataloader,
            train_loss_record=[],val_loss_record=[],f1_score_record=[],epoch_record=0):
    
  scaler=GradScaler() # USING AVERAGE-MIXED-PRECISION
  save_best_model = SaveBestModel()
  early_stopping=Early_Stopping(patience=config['early_stopping_patience'],min_delta=config['early_stopping_delta'])
  
  writer=SummaryWriter('/home/runs/squad')
  total_iteration=0

  for epoch in range(epoch_record,epochs):

    # dataloader and shuffling must be added here for checkpointing
    
    tq=tqdm(train_dataloader,colour='green')
    tq.set_description('epoch %s' %(epoch+1))
    running_loss_per_epoch=0

    for context,question,answer,start,end in tq:
      
      model.train()
      current_batch_size=context.size(0)
      
      context,question,answer,start,end=context.to(device),question.to(device),answer.to(device),start.to(device),end.to(device)
      
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

      if (total_iteration+1)%config['model_checkpointing_iteration']==0:
        
        with torch.no_grad():
          
          model.eval()
          val_loss=validation_loss(dt_val=val_dataloader,model=model)
          val_loss_record.append(val_loss)
          writer.add_scalar('validation_loss',val_loss,(total_iteration//config['model_checkpointing_iteration'])+1)

          if early_stopping is True :
            print('EARLY STOPPING ACTIVATED')  
            break

        save_best_model(val_loss,epoch,model,optimizer,train_loss_record,val_loss_record,f1_score_record) # checkpointing low val_loss model

      total_iteration+=1     

      tq.set_postfix(loss=loss_) # tqdm printing of loss

    print('running loss for epoch {}: '.format(epoch+1), running_loss_per_epoch/len(train_dataloader.dataset)) # loss per epoch avg
    
    with torch.no_grad():
      f1=validate_f1(dt=val_dataloader,models=model)
      f1_score_record.append(f1)
      print(" validation f1_score for epoch {}".format(epoch+1),f1)

      writer.add_scalar('f1_score/epoch',f1,epoch+1)

  writer.close()
  
  torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, 'final_model_150.pth') # final sav  

  save_graphs(train_loss_record,val_loss_record,f1_score_record)

#-------------------------------------------------------------------------------------------------

if __name__=='__main__':
  pass    
