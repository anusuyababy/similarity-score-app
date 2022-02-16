# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 19:33:20 2022

@author: kennedy
"""

from flask import Flask, render_template, request
from strsimpy.cosine import Cosine
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    CLO = request.form['CLO']
    PLO = request.form['PLO']
    cosine = Cosine(2)
    sen1 = cosine.get_profile(CLO)
    sen2 = cosine.get_profile(PLO)
    cosine_sim = cosine.similarity_profiles(sen1, sen2) 
    return render_template('index.html', prediction='The similarity score for the given CLO and PLO is {}.'.format(cosine_sim))

if __name__=="__main__":
    app.run(debug=True)
