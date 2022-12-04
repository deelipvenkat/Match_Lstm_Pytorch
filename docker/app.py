#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from spacy.lang.en import English
from flask import Flask, request, jsonify, render_template
import pickle
import torch
from models import load_model
from input_pipelines import input_processing,proba_score_search

"""
installations necessary - torch-cpu , torchtext-maybe , spacy
"""

app = Flask(__name__)
app._static_folder="/home/deelip/Desktop/squad/deployment/static"

vocab=pickle.load(open('vocabulary.pkl','rb'))
trained_weights=pickle.load(open("trained_weights.pkl","rb"))

model=load_model(hidden_units=150,device='cpu',weights=trained_weights)
saved=torch.load('model.pth',map_location='cpu')
model.load_state_dict(saved['model_state_dict'])
model.eval()
nlp=English()

del saved

@app.route('/')
def home():
    return render_template('front_end.html')


@app.route('/predict',methods=['POST'])
def predict():

    if request.method=='POST':

        s1,s2 = request.form.get("context_area"),request.form.get("question_area")

        c,q,tok_c=input_processing(s1,s2,vocab)
        attn_matrix=model(c,q)
        score=proba_score_search(attn_matrix).squeeze(0)
        answer= "".join([word.text_with_ws for word in tok_c[score[0]:score[1]+1]])    

    return render_template('front_end.html', prediction_text= answer)


if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0',port=16003)
