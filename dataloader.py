"""set of dataloaders,dataset,collate functions, data/batch sampler classes """

import torch
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pad_sequence
import random
from torch.utils.data.sampler import BatchSampler
import math
from torch.utils.data.sampler import SubsetRandomSampler
from torchnlp.samplers.sorted_sampler import SortedSampler
from torchnlp.utils import identity
from torch.utils.data import RandomSampler,SequentialSampler
from torchnlp.random import set_seed

#*************************************************************************************************

class squad_index_data(Dataset):
  def __init__(self,context,question,answer,start,end):
    super().__init__()
    self.context=context
    self.question=question
    self.answer=answer
    self.start=start
    self.end=end
    self.num_samples=self.context.shape[0]

  def __getitem__(self,index):
    return self.context[index],self.question[index],self.answer[index],self.start[index],self.end[index]

  def __len__(self):
    return self.num_samples

#*************************************************************************************************

# WITHOUT START AND END INDEX

class squad_data(Dataset):
  def __init__(self,context,question,answer):
    super().__init__()
    self.context=context
    self.question=question
    self.answer=answer
    self.num_samples=self.context.shape[0]

  def __getitem__(self,index):
    return self.context[index],self.question[index],self.answer[index]

  def __len__(self):
    return self.num_samples

#*************************************************************************************************

class retrain_dataset(Dataset):
  def __init__(self,context,question,start,end):
    super().__init__()
    self.context=context
    self.question=question
    self.start=start
    self.end=end
    self.num_samples=len(context)

  def __getitem__(self,index):
    return self.context[index],self.question[index],self.start[index],self.end[index]

  def __len__(self):
    return self.num_samples


#*************************************************************************************************

def batched_indexed_data(c_pad,q_pad,ans_pad,ans_start,ans_end
    ,batch,num_workers,shuffle=False):
  
  squad_train=squad_index_data(c_pad,q_pad,ans_pad,ans_start,ans_end)
  data_train=DataLoader(dataset=squad_train,batch_size=batch,
    pin_memory=True,shuffle=shuffle,num_workers=num_workers)

  return data_train

def batched_data(c_pad,q_pad,ans_pad,batch,num_workers,shuffle=True):
  
  squad_train=squad_data(c_pad,q_pad,ans_pad)
  data_train=DataLoader(dataset=squad_train,batch_size=batch,pin_memory=True,shuffle=shuffle,num_workers=num_workers)

  return data_train

#*************************************************************************************************


def custom_collate(batch):
    a=list(zip(*batch))
    t=pad_sequence(a[0],batch_first=True)
    t2 = torch.stack(a[1], 0)        
    return t,t2,a[2],a[3]

def raw_collate(batch):
    a=list(zip(*batch))
    t=pad_sequence(a[0],batch_first=True)
    t2 = torch.stack(a[1], 0)        
    return t,t2,a[2],a[3]


#**************************************************************************************************

class BucketBatchSampler(BatchSampler):

    def __init__(self,
                    sampler,
                    batch_size,
                    drop_last,
                    sort_key=lambda i: i,
                    bucket_size_multiplier=100):
        super().__init__(sampler, batch_size, drop_last)
        self.sort_key = sort_key
        self.bucket_sampler = BatchSampler(sampler,
                                            min(batch_size * bucket_size_multiplier, len(sampler)),
                                            False)

    def __iter__(self):
        for bucket in self.bucket_sampler:
            sorted_sampler = SortedSampler(bucket, self.sort_key)
            for batch in SubsetRandomSampler(
                    list(BatchSampler(sorted_sampler, self.batch_size, self.drop_last))):
                yield [bucket[i] for i in batch]

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)

#*************************************************************************************************



if __name__=='__main__':

  pass  