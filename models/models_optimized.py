"""module containing the match-lstm model optimized with pack-padded-sequence"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.utils.rnn import pad_sequence,pad_packed_sequence,pack_padded_sequence
from utils import seq_length
# CONTEXT LSTM
device=['cuda' if torch.cuda.is_available() is True else 'cpu'][0]

with open('/home/config.yaml') as f:
  config=yaml.safe_load(f)

#-------------------------------------------------------------------------------------------------

class context_lstm(nn.Module):

  def __init__(self,hidden_units,trained_weights,embed_size=config['embed_size'],bidirectional=False):
    super().__init__()
    self.hidden_units=hidden_units
    self.embed_size=embed_size
    self.bidirectional=bidirectional
    self.trained_weights=trained_weights

    self.embedding_layer=nn.Embedding.from_pretrained(self.trained_weights,freeze=True)
    self.lstm_layer=nn.LSTM(self.embed_size,self.hidden_units,batch_first=True,bidirectional=bidirectional)

  def forward(self,x,hidden1,hidden2): # set x as a tuple containing context/seq_length
    x=self.embedding_layer(x)

    context_seq=seq_length(x) # finding seq length

    packed=pack_padded_sequence(x,lengths=context_seq,enforce_sorted=False,batch_first=True)
    
    output,_=self.lstm_layer(packed,(hidden1,hidden2))
    output_1,_=pad_packed_sequence(output,batch_first=True)

    return output_1
  
  def init_hidden(self,batch):
    return torch.zeros(1,batch,self.hidden_units,device=device),torch.zeros(1,batch,self.hidden_units,device=device)

#-------------------------------------------------------------------------------------------------
# QUESTION LSTM

class question_lstm(nn.Module):

  def __init__(self,hidden_units,trained_weights,embed_size=config['embed_size'],bidirectional=False):
    super().__init__()
    self.hidden_units=hidden_units
    self.embed_size=embed_size
    self.bidirectional=bidirectional
    self.trained_weights=trained_weights

    self.embedding_layer=nn.Embedding.from_pretrained(self.trained_weights,freeze=True)
    self.lstm_layer=nn.LSTM(self.embed_size,self.hidden_units,batch_first=True,bidirectional=bidirectional)

  def init_hidden(self,batch):
    return torch.zeros(1,batch,self.hidden_units,device=device),torch.zeros(1,batch,self.hidden_units,device=device)

  def forward(self,x,hidden1,hidden2):
    x=self.embedding_layer(x)
    
    output,_=self.lstm_layer(x,(hidden1,hidden2))
    # ADDING DROPOUT HERE *******************  
  
    return output

#-------------------------------------------------------------------------------------------------
# MATCH ATTENTION LAYER

class attention_1(nn.Module):
  def __init__(self,hidden_units):
    
    super().__init__()
    self.hidden_units=hidden_units
    
    self.linear_g2=nn.Linear(self.hidden_units,self.hidden_units)
    self.linear_g3=nn.Linear(self.hidden_units,self.hidden_units,bias=False)
    self.linear_g1=nn.Linear(self.hidden_units,self.hidden_units,bias=False)

    self.linear_g4=nn.Linear(self.hidden_units,1,bias=True)# near softmax operation    


  def forward(self,context,question,hidden):

    attn_weights=self.score(context,question,hidden)
    
    attn_weights_1=attn_weights.transpose(1,2) # batch*1*question_length
    
    attn_combination=torch.bmm(attn_weights_1,question) # shape= batch*1*hidden_units = 34,1,100
    
    return attn_combination  # returns a bmm product of attention weights & question hidden states.

  def score(self,context,question,hidden):
    output_g3=self.linear_g3(hidden.transpose(0,1)) # g3 operation

    output_g2=self.linear_g2(context).unsqueeze(1) # g2 operation

    output_g1=self.linear_g1(question) # g1 operation

    output_=output_g2+output_g3 # sum of g3 and g2

    d_=output_g1.size(1) # computing batch_size , hidden dimension and seq-length
    batch=output_.size(0)
    hid_dim=output_.size(2)

    output_repeat=output_.repeat(1,d_,1).view(batch,hid_dim,-1)# outer product of G2 matrix

    output_g=output_repeat+output_g1.view(batch,hid_dim,-1) # sum of g1 and G2 MATRIX

    output_1=torch.tanh(output_g) # tanh operation
    
    output_final=self.linear_g4(output_1.view(batch,-1,hid_dim)) # g4 operation
    
    attn_weights=F.softmax(output_final,dim=1) # SOFTMAX OVER OUTPUTS

    return attn_weights # shape = batch*question_timestamps*1

#-------------------------------------------------------------------------------------------------
attention_common=attention_1(config['hidden_units']) # added this part to handle common attention parameters.
# doubts regarding whether this attention will be update twice due to usage in 2 different
# networks or is this the ideal implementation.


# MATCH DECODER-FORWARD LSTM

class decoder_forward(nn.Module):
  def __init__(self,hidden_units):
    super().__init__()
    self.hidden_units=hidden_units
    self.attention=attention_common #attention_1(self.hidden_units)
    self.lstm_layer=nn.LSTM(self.hidden_units*2,self.hidden_units,bidirectional=False,batch_first=True)

  def forward(self,context,question,hidden,mem):

    attn_combination=self.attention(context,question,hidden)  # context shape= (batch,1,hidden_units)
        
    context=context.unsqueeze(1)
    

    output,(_,mem)=self.lstm_layer(torch.cat((context,attn_combination),dim=2),(hidden,mem)) 

    return output,mem

# MATCH DECODER-BACKWARD LSTM

class decoder_reverse(nn.Module):
  def __init__(self,hidden_units):
    super().__init__()
    self.hidden_units=hidden_units
    self.attention= attention_common #attention_1(self.hidden_units)
    self.lstm_layer=nn.LSTM(self.hidden_units*2,self.hidden_units,bidirectional=False,batch_first=True)

  def forward(self,context,question,hidden,mem):

    attn_combination=self.attention(context,question,hidden)  # context shape= (batch,1,hidden_units)
        
    context=context.unsqueeze(1)
    
    output,(_,mem)=self.lstm_layer(torch.cat((context,attn_combination),dim=2),(hidden,mem)) 

    return output,mem
#-------------------------------------------------------------------------------------------------
# ANSWER POINTER ATTENTION LAYER

class attention_2(nn.Module):
  def __init__(self,hidden_units):
    super().__init__()
    self.hidden_units=hidden_units
    
    self.linear_g2=nn.Linear(self.hidden_units,self.hidden_units,bias=True)

    self.linear_g1=nn.Linear(self.hidden_units*2,self.hidden_units,bias=False)

    self.linear_g3=nn.Linear(self.hidden_units,1,bias=True)# near softmax operation    


  def forward(self,match_output,hidden):

    attn_weights=self.score(match_output,hidden)
    
    attn_weights_1=attn_weights.transpose(1,2) # batch*1*question_length
    
    attn_combination=torch.bmm(attn_weights_1,match_output) # shape= batch*1*hidden_units = 34,1,100
    
    return attn_weights_1,attn_combination  # returns a bmm product of attention weights & question hidden states.

  def score(self,match_output,hidden):
    output_g2=self.linear_g2(hidden).transpose(0,1) # g3 operation

    output_g1=self.linear_g1(match_output) # g2 operation

    d_=output_g1.size(1) # computing batch_size , hidden dimension and seq-length
    #batch=output_g1.size(0)
    #hid_dim=output_g1.size(2)

    output_repeat=output_g2.repeat(1,d_,1) # outer product
    
    output_g=output_repeat+output_g1 # sum of g1 and G2 MATRIX

    output_1=torch.tanh(output_g) # tanh operation

    output_final=self.linear_g3(output_1) # g4 operation

    attn_weights=F.softmax(output_final,dim=1) # SOFTMAX OVER OUTPUTS
    
    return attn_weights # shape = batch*question_timestamps*1

#-------------------------------------------------------------------------------------------------
# ANSWER POINTER DECODER LSTM

class decoder_2(nn.Module):
  def __init__(self,hidden_units):
    super().__init__()
    self.hidden_units=hidden_units
    self.attention_2=attention_2(self.hidden_units)
    self.lstm_layer_2=nn.LSTM(self.hidden_units*2,self.hidden_units,bidirectional=False,batch_first=True)

  def forward(self,match_output,hidden,mem):

    attn_weight,attn_combination=self.attention_2(match_output,hidden)  # context shape= (batch,1,hidden_units)
    
    output,(_,mem)=self.lstm_layer_2(attn_combination,(hidden,mem))

    return output,mem,attn_weight

#-------------------------------------------------------------------------------------------------
#MATCH LSTM WITH ANSWER POINTER NETWORK

class match_lstm(nn.Module):
  def __init__(self,encoder_c,encoder_q,decoder_forward,decoder_reverse,decoder_ans):
    super().__init__()
    self.encoder_c=encoder_c
    self.encoder_q=encoder_q
    self.decoder_forward=decoder_forward
    self.decoder_reverse=decoder_reverse
    self.decoder_ans=decoder_ans
  
  def forward(self,context,question): # context seq is spacy tokenized length.

    if context.ndim==1:
      context,question=context.unsqueeze(0),question.unsqueeze(0)
    
    batch=question.size(0) # batch size of data
    
    h1,h2=self.encoder_c.init_hidden(batch) # initialising hidden/mem states for encoder question/context lstm
    h3,h4=self.encoder_q.init_hidden(batch)

    context_hidden=self.encoder_c(context,h1,h2) # getting context/question pre-processing layers output.
    question_hidden=self.encoder_q(question,h3,h4)

    hidden_size=context_hidden.size(2)  # hidden units of pre-processing lstm outputs.
    timesteps=context_hidden.size(1)    # timesteps in match-lstm- handles pad seq , context -P=400

    hidden_set_f=torch.empty(batch,timesteps,hidden_size,device=device) # initialising desired final output
    hidden_set_r=torch.empty(batch,timesteps,hidden_size,device=device)
    
    hidden_f,mem_f=torch.zeros(1,batch,hidden_size,device=device),torch.zeros(1,batch,hidden_size,device=device) # initialising 1st hidden/mem states for match-lstm
    hidden_r,mem_r=torch.zeros(1,batch,hidden_size,device=device),torch.zeros(1,batch,hidden_size,device=device)

    # context=([34, 400, 100]) question=([34, 30, 100]) -> hidden shapes

    for t in range(timesteps):

      hidden_f,mem_f=self.decoder_forward(context=context_hidden[:,t,:],question=question_hidden,hidden=hidden_f,mem=mem_f) 
      hidden_r,mem_r=self.decoder_reverse(context=context_hidden[:,(timesteps-t-1),:],question=question_hidden,hidden=hidden_r,mem=mem_r) 
      # need only hidden/mem states for loop
      
      hidden_=hidden_f.squeeze(1)
      hidden_f=hidden_f.transpose(0,1)
      hidden_set_f[:,t,:]=hidden_ # adding hidden state values for each timestep batch wise

      hidden_1=hidden_r.squeeze(1)
      hidden_r=hidden_r.transpose(0,1)
      hidden_set_r[:,(timesteps-t-1),:]=hidden_1 # adding hidden state values for each timestep batch wise


    hidden_match= torch.cat((hidden_set_f,hidden_set_r),dim=2) # shape -B*T*(2*hidden)
    
    # convert this to constant size of 400 --> b*400*(2*hidden)
    missing_values=400-timesteps
    resizer=torch.zeros((batch,missing_values,config['hidden_units']*2),device=device)
    hidden_match=torch.cat((hidden_match,resizer),dim=1)     
    # setting the output of all batches to same shape
    

    hidden_ans,mem_ans=torch.zeros(1,batch,hidden_size,device=device),torch.zeros(1,batch,hidden_size,device=device)
    attn_matrix=torch.empty(batch,config['context_length'],2,device=device)

    for j in range(2):  # 2 is used considering the 2 outputs for start and end index
      
      hidden_ans,mem_ans,attn_weights=self.decoder_ans(hidden_match,hidden=hidden_ans,mem=mem_ans) # attn shape- batch ,1 ,timesteps

      attn_weights=attn_weights.transpose(1,2)
      attn_weights=attn_weights.squeeze(2)
      
      hidden_ans=hidden_ans.transpose(0,1)
      attn_matrix[:,:,j]=attn_weights # adding hidden state values for each timestep batch wise

    return attn_matrix

#-------------------------------------------------------------------------------------------------
def weights_init(model):
  """
  initializing weights of model lstm and linear layers to xavier normal.
  """
  for name , parameter in model.named_parameters():
    if ('weight' in name and 'lstm' in name) or ('weight' in name and 'linear' in name):
        nn.init.xavier_normal_(parameter) 


def load_model(hidden_units,device,weights,xavier_init=True):
  
  context_preprocess=context_lstm(hidden_units,trained_weights=weights).to(device) # context lstm
  question_preprocess=question_lstm(hidden_units,trained_weights=weights).to(device) # question lstm
  de_f=decoder_forward(hidden_units).to(device) # foward match decoder
  de_b=decoder_reverse(hidden_units).to(device) # reverse match decoder
  de_ans=decoder_2(hidden_units).to(device) # answer pointer decoder

  match_model=match_lstm(encoder_c=context_preprocess,encoder_q=question_preprocess,decoder_forward=de_f,decoder_reverse=de_b,decoder_ans=de_ans).to(device)       
  
  if xavier_init==True:
    weights_init(match_model)

  return match_model

if __name__=='__main__':
  pass

