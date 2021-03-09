#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold as sKFold
import sklearn
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
from sklearn.metrics import f1_score as f1

import os
import sys
import glob
import json
import pickle
filenames = glob.glob(os.path.join("data", "tt*.pkl"))
all_pred =[]
title =[]
ground_truth = []
movie_no = []
for i, file in enumerate(filenames):
    movie = pickle.load(open(file, "rb"))
    g_truth = movie['scene_transition_boundary_ground_truth']
    encoder = LabelEncoder()
    encoder.fit(g_truth)
    encoded_gtruth = encoder.transform(g_truth)
    init_pred = movie['scene_transition_boundary_prediction'].numpy()
    all_pred.extend(init_pred)
    length = len(init_pred)
    a = [movie['imdb_id']]
    num = np.full(length, i)
    sno = a*length
    title.extend(sno)
    movie_no.extend(num)
    ground_truth.extend(encoded_gtruth)
all_movies = pd.DataFrame(list(zip(movie_no,title, all_pred)), columns =['movie_no','id', 'prob'])     
y_pred = []
y_index = []
kf = sKFold(n_splits=4)
ground_truth = np.array(ground_truth)
for train_index, test_index in kf.split(all_movies[['movie_no','prob']],ground_truth):
    train_len = len(train_index)
    test_len = len(test_index)
    X_train, X_test = all_movies[['movie_no','prob']].iloc[train_index,:], all_movies[['movie_no','prob']].iloc[test_index,:]
    y_train, y_test = ground_truth[train_index], ground_truth[test_index]
    model = Sequential()
    model.add(Dense(5, input_dim= 2,activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(lr = 0.1)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics = ['accuracy'])
    print(model.summary())
    class_weight = {0:1,1:9}
    history = model.fit(X_train, y_train,batch_size=2, epochs=15,verbose = 0,class_weight = class_weight)
    y_predict = model.predict(X_test)
    y_pred.extend(y_predict)
    y_index.extend(test_index)
prediction_df = pd.DataFrame(list(zip(y_index,y_pred)),columns = ['y_index','y_pred'])
prediction_df = prediction_df.sort_values('y_index')
prediction_df['id'] = all_movies['id']
prediction_df['real'] = ground_truth
for movie in list(set(prediction_df.id)):
    org_movie = pickle.load(open((movies + ".pkl"), "rb"))
    org_movie['scene_transition_boundary_prediction'] = prediction_df.loc[prediction_df['id'] == movie]['y_pred']
    org_movie['scene_transition_boundary_ground_truth'] = org_movie['scene_transition_boundary_ground_truth'].numpy()
    org_movie['shot_end_frame'] = movie['shot_end_frame'].numpy()
    with open((movie + ".pkl"), 'wb') as fn:
        pickle.dump(org_movie, fn, protocol=pickle.HIGHEST_PROTOCOL)

