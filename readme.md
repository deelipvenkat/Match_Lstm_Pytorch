
442 titles in train , 48 in dev , 46 in test - total 536 titles
given 23215 paragraphs - total questions - 87599
31 repeating questions in test set.

tokenized text contains in total 11.6 million tokens - if pack sequence not used - 33.2 million
we can achieve theoretical efficiency i.e reduction in computations by 65%.avg tokens/context -140

entirely different titles in train/dev/test 

preprocessing - start & end tokens for answers, unk tokens for some words, word embedding to use
integer representation of words , special char/lowercasing should not be removed, tokenizer ,split train and val set

PREPROCESSING:
encoding special characters & numbers
['unk'],['pad']- addition in the vocab and setting a random distant vectors.
splitting of train and validation set -easy , setting a prototype set
char to int conversion of text
padding & truncating to the desired values.(400,30,15/20)

multiple answers in text can be dealt with later -check with all the answers & consider the highest score.

MODEL ARCHITECTURE: ATTENTION ENCODER DECODER FOR QUESTION ANSWERING

the paper has mentioned there is still some overfitting in the model, try using dropout or some reguralization/constraints

further improvements :

pad values probabilites affect loss to some extent since sum of all values =1 , so they skew the 
probabilites to a certain extent.

pad index change-files : metrics , evaluation,utils

COLAB NOTES :-

FINAL LOSS FOR HERSHED - 3.23 after 8 epochs(maybe train loss). VAL LOSS

I have put a large learning rate leading to poor result & loss shootup in the end.

fixed match-lstm attention issue

1000 outputs with start> end(~817) , end-start(~580) IN TOTAL 1397~ 13.21%

understand bi-ans pointer- got it - same attn operation , 2 hidden states - one for forward operation and one for backward operation. wt about attn weights. no sense of direction given for lstm still learnt mapping using backprop , so maybe if i give different inputs to bi-lstm & take average of the 2 sets of attn matrices and train, maybe it will learn that mapping & will learn better.280k parameters increase with bi-ans pointer.

EVEN CURRENT SETTINGS WITH BI-ANS-POINTER SEEM GOOD

223 contexts above 400 pad limit
ONLY DESIGNATIONS ARE CHANGING , no other edge cases while spacy normal casing

there are very few, for above 400 answer is below 400 index, so no problem. max -23 incorrectly labelled datapoints.

***************************************************************************************************
TRAINING TIME/BATCH SIZE

32  - 15hrs = 10 HRS
64  - 11.5hrs  -- 5.2 hrs with A100
128 - 6.5hrs
256 - 3 hrs
512 - 1.33 hrs

can try for 128 with early stopping

***************************************************************************************************
achieved f1 score=37% & EM score of 6% - for a model trained for 5 epochs & 100 hidden dimensions
on the prototype dataset which is a part of train set.

prototype set- f1-score = 38.5% , EM-score = 17.65% after fixing end index from +1 +1 to +1

val set -  f1-score=34.36% EM-score= 22.32% initial 5epoch model.

ON TEST-SET -VERY LONG INFERENCE TIME - REAL UNSEEN DATASET
F1_score:  37.85%
EM_score:  27.12%

ON TEST-SET 10 EPOCHS , LR=0.005 , BATCH SIZE=256 , HID_DIM=150 Without checkpointing
F1_score:  51.4%
EM_score:  38.5%  

On train set for best model - f1: 75%  EM : 60% -THERE IS OVERFITTING.

validation loop on val set - 92 sec - for batch size 32

NEW RECORD -YEAHHH-lr = 0.005 TRUE

same settings - best saved model results - as we can see , not much overfitting.
dropout seems like unnecessary - with checkpointing.

TEST SET                             TRAIN SET              VALIDATION SET
F1_SCORE:  63.04%                    F1_SCORE:  70%         F1_SCORE:  58.68%
EM_SCORE:  50.43%                    EM_SCORE:  53%         EM_SCORE:  41%    

highest model - val loss= 4.1813 , f1 =58.69%

NEW RECORD-OH YEAHH

ANSWER_SPAN=20
TEST SET                             TRAIN SET               VALIDATION SET
F1_SCORE:  65.32%                    F1_SCORE:  71.8%        F1_SCORE:  60.93%
EM_SCORE:  52.46%                    EM_SCORE:  54.6%        EM_SCORE:  42.5%   

ANSWER_SPAN=15 LR=0.005
TEST SET                             TRAIN SET               VALIDATION SET
F1_SCORE:  65.45% ***                F1_SCORE:  71.93%       F1_SCORE:  61.01%
EM_SCORE:  52.66%                    EM_SCORE:  54.72%       EM_SCORE:  42.76%  

model seems to be overfitting on 300-D word embedding min loss at 5 for batch size 32 & min loss of 5.5 for batch size 256 before loss exploding upward again.

USING 300d word embeddings & batch size=64 & learning rate = 0.005

TEST SET
F1-SCORE: 52.1%
EM-SCORE: 39.32%

YEAH BOY - after fixing answer start/end index lr=0.002
TEST SET                VAL SET                   
F1-SCORE:  67.61%       F1-SCORE:  61.93%
EM-SCORE:  54.19%       EM-SCORE:  43.6%

TRAIN SET
F1-SCORE:  72.73%
EM-SCORE:  55.22%

with adamax - test[f1=60% em=45%] slow convergence due to low learning rate

with lr=0.0015 , after using spacy tokens
TEST SET               VAL SET 
F1-SCORE:67.16%        F1-SCORE:61.48%   
EM-SCORE:53.97%        EM-SCORE:42.69% 

TRAIN SET
F1-SCORE:72.76%
EM-SCORE:55.61%

WITH RETRAINING - HIGH SCORE lr=0.002 val_loss=3.2625
TEST SET                VAL SET                   
F1-SCORE:  67.77%       F1-SCORE:  62.66%
EM-SCORE:  55.08%       EM-SCORE:  44.17%

TRAIN SET
F1-SCORE:  72.54%
EM-SCORE:  55.56%


batch_size=128
TEST SET                VAL SET
F1-SCORE:  67.88        F1-SCORE:61.64%
EM-SCORE:  54.73        EM-SCORE:43.7%

TRAIN SET
F1-SCORE: 71.57 %
EM-SCORE: 54.57%


***************************************************************************************************

GOAL : F1-SCORE > 60%    EM-SCORE > 50% -TARGET REACHED

crossed 65% mark - next 70%

PUSH IT TOWARDS (70%,60%)

***************************************************************************************************
UPGRADES 

using batch size of 32/64 for training but takes a lot of time to train.

unk token , scalars,vectors,answer ptr hidden state initialization

weight decay/grad clipping/batch-norm/Dropout

***************************************************************************************************

***************************************************************************************************
TO-DO :

bottleneck profiler-code optimization.

ensemble deep learning models.

DEPLOYMENT

VERY MUCH LATER:

attention weights visualization(final model)

performance across different types of questions[how,who,when,why,what,which,where](final model)

performance across different answer lengths.(final model)

**************************************************************************************************


**************************************************************************************************
HYPOTHESIS TESTING

does using exponential masking function speeds up convergence and also leads to better convergence , also think about using masking in match lstm hidden state.
Can verify this by saving pad tokens attention during training for every few iterations.

gradients are working fine in match attention layer except , grad values are always near 0 for attention linear layer(g1)- check others

exponential masking - excluding all the pad values or setting the value to zero- setting to some specific value will again become a hyperparameter leading to unknown results.
So I think full exponential masking and excluding of pad values is the way to go.
is it beneficial - faster convergence vs longer iteration runtime
**************************************************************************************************

model optimization- 
changes in dataloader - adding seq_length of context/question inside model
collate fn to pad/truncate length as per batch
creating a sorted custom dataloader.

deployment - 
vocab -50mb, trained_weight- 72mb , spacy tokenizer/pytorch , model- 50mb 

seems like my validation loss is not representative of f1-score.
