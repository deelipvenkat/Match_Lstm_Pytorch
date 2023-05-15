# **Question Answering with Match-LSTM on SQuAD 1.1** 

Re-Implementation of [MATCH-LSTM ( *Shuohang Wang & Jing Jiang 2016* )](https://arxiv.org/abs/1608.07905) model on [SQUAD V1.1 ( *Rajpurkar et al. 2016* )](https://arxiv.org/abs/1606.05250) dataset on Pytorch.

## Model Architecture

![match lstm architecture](/images/match_lstm.png)



## Requirements 

Create a conda environment for python(3.8.12) and install the dependencies used in the project using the command

```
pip install -r requirements.txt
```

To run the model in your local machine, Follow these instructions to download & run the docker image of the model.

```
# Downloading the docker-image of match-lstm model from dockerhub
docker pull deelipvenkat/squad_image:pytorch

# starting a docker container from the image
docker run -d -p 6666:16003 deelipvenkat/squad_image:pytorch
```
In your web-browser, go to this url http://localhost:6666/ to use the application

## Model & Training Details

- Learning rate: 0.007  
- Batch size: 64  
- Hidden size: 150  
- Word Embedding: Glove 100d  
- Optimizer: Adam
- Epochs: 8  
- Training Time: ~ 8hrs  
- Training Hardware: Nvidia RTX 5000 16gb VRAM


## Results

| Model        | F1-SCORE | EM-SCORE|
|--------------|----------|---------|
|Match-lstm(b) | 68.66%   | 55.54%  | 

* (b) refers to the boundary model in match-lstm, where the start/end index of the answer in the passage is predicted.
* The model was evaluated on squad v1.1 dev set.
## References

[Wang, S., & Jiang, J. (2016). Machine comprehension using match-lstm and answer pointer. arXiv preprint arXiv:1608.07905.](https://arxiv.org/abs/1608.07905)

[Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). SQuAD: 100,000+ questions for machine comprehension of text. arXiv preprint arXiv:1606.05250.](https://arxiv.org/abs/1606.05250)
