#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from tensorflow.keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold as sKFold
from sklearn.preprocessing import StandardScaler
import sklearn
from tensorflow.keras.callbacks import EarlyStopping



def df_from_file(data):
    place_diff = []
    for i in range(len(data['place'])-1):
        sub = data['place'][i+1] - data['place'][i]
        place_diff.append(sub.numpy())

    cast_diff = []
    for i in range(len(data['cast'])-1):
        sub = data['cast'][i+1] - data['cast'][i]
        cast_diff.append(sub.numpy())

    action_diff = []
    for i in range(len(data['action'])-1):
        sub = data['action'][i+1] - data['action'][i]
        action_diff.append(sub.numpy())

    audio_diff = []
    for i in range(len(data['audio'])-1):
        sub = data['audio'][i+1] - data['audio'][i]
        audio_diff.append(sub.numpy())

    df = pd.DataFrame(list(zip(place_diff, cast_diff, action_diff, audio_diff, data['scene_transition_boundary_prediction'].numpy())), columns =['place', 'cast','action','audio','prediction']) 
    return(df)

def scale_df(df):
    df.place = [sklearn.preprocessing.minmax_scale(item, feature_range=(0, 1), axis=0, copy=True) for item in df.place]
    df.cast = [sklearn.preprocessing.minmax_scale(item, feature_range=(0, 1), axis=0, copy=True) for item in df.cast]
    df.action = [sklearn.preprocessing.minmax_scale(item, feature_range=(0, 1), axis=0, copy=True) for item in df.action]
    df.audio = [sklearn.preprocessing.minmax_scale(item, feature_range=(0, 1), axis=0, copy=True) for item in df.audio]
    return(df)

def clean(df):
    for i in range(len(df)):
        df.place[i] =  df.place[i][~np.isnan(df.place[i])]
        df.cast[i] =  df.cast[i][~np.isnan(df.cast[i])]
        df.action[i] =  df.action[i][~np.isnan(df.action[i])]
        df.audio[i] =  df.audio[i][~np.isnan(df.audio[i])]
    return(df)

def merge_all_features(df):    
    all_features = []
    for i in range(len(df)):
        a = df['place'][i].tolist()
        a.extend(df['cast'][i])
        a.extend(df['action'][i])
        a.extend(df['audio'][i])
        a.append(df['prediction'][i])
        a = np.array(a)
        all_features.append(a) 
    return(all_features)

def flatten_df (all_features):    
    df = pd.DataFrame(all_features[0].reshape(1,-1))
    for i in range(len(all_features)-1):
        df = df.append(pd.DataFrame(all_features[i+1].reshape(1,-1)), ignore_index=True)
    return df

if __name__ == "__main__":
    import os
    import sys
    import glob
    import json
    import pickle
    filenames = glob.glob(os.path.join("data", "tt*.pkl"))
    all_movies =  pd.DataFrame()
    g_truth =[]
    for i, file in enumerate(filenames):
        movie = pickle.load(open(file, "rb"))
        grnd_truth = movie['scene_transition_boundary_ground_truth']
        encoder = LabelEncoder()
        encoder.fit(grnd_truth)
        encoded_gtruth = encoder.transform(grnd_truth)
        movie_df = df_from_file(movie)
        scaled_movie_df = scale_df(movie_df)
        clean_df = clean(scaled_movie_df)
        all_features = merge_all_features(clean_df)
        final_df = flatten_df(all_features)
        length = len(final_df)
        a = [movie['imdb_id']]
        sno = a*length
        final_df['id'] = sno
        num = np.full(length, i)
        final_df['movie_no'] = num
        all_movies = all_movies.append(final_df)
        g_truth.extend(encoded_gtruth)
    y_pred = []
    y_index = []
    kf = sKFold(n_splits=4)
    for train_index, test_index in kf.split(all_movies.loc[:, all_movies.columns != 'id'],gtruth):
        train_len = len(train_index)
        test_len = len(test_index)
        X_train, X_test = (all_movies.loc[:, all_movies.columns != 'id']).iloc[train_index,:], (all_movies.loc[:, all_movies.columns != 'id']).iloc[test_index,:]
        y_train, y_test = gtruth[train_index], gtruth[test_index]
        X_train = X_train.values.reshape(train_len,3586,1)
        X_test = X_test.values.reshape(test_len,3586,1)
        model = Sequential()
        model.add(LSTM(50, input_shape = (3586,1),activation = 'sigmoid',return_sequences=True,dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        class_weight = {0:1,1:9}
        history = model.fit(X_train, y_train,batch_size=1, epochs=30,validation_split=0.1,class_weight = class_weight,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
        y_predict = model.predict(X_test)
        y_pred.extend(y_predict)
        y_index.extend(test_index)
    prediction_df = pd.DataFrame(list(zip(y_index,y_pred)),columns = ['y_index','y_pred'])
    prediction_df = prediction_df.sort_values('y_index')
    prediction_df['given_pred'] = all_movies[3584]
    prediction_df = prediction_df.loc[:, prediction_df.columns != 'y_index']
    prediction_df['id'] = all_movies['id']
    prediction_df['real'] = ground_truth
    for movie in list(set(prediction_df.id)):
        org_movie = pickle.load(open((movies + ".pkl"), "rb"))
        org_movie['scene_transition_boundary_prediction'] = prediction_df.loc[prediction_df['id'] == movie]['y_pred']
        org_movie['scene_transition_boundary_ground_truth'] = org_movie['scene_transition_boundary_ground_truth'].numpy()
        org_movie['shot_end_frame'] = movie['shot_end_frame'].numpy()
        with open((movie + ".pkl"), 'wb') as fn:
            pickle.dump(org_movie, fn, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:




