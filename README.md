# **MACHINE QUESTION-ANSWERING SYSTEM** 

Re-Implementation of [MATCH-LSTM ( *Shuohang Wang & Jing Jiang 2016* )](https://arxiv.org/abs/1608.07905) model on [SQUAD V1.1 ( *Rajpurkar et al. 2016* )](https://arxiv.org/abs/1606.05250) dataset on Pytorch.


## Requirements 

Create a conda environment for python(3.8.12) and install the dependencies used in the project using the command

```
pip install -r requirements.txt
```
## Execution

## Results

| Model        | F1-SCORE | EM-SCORE|
|--------------|----------|---------|
|Match-lstm(B) | 68.66%   | 55.54%  | 

* (B) refers to the boundary model in match-lstm, where the start & end index of the answer tokens is predicted.
## References
