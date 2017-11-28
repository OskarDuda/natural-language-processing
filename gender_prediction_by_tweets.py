#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 19:45:02 2017

@author: oskar
"""

import string
import numpy as np
import pandas as pd
from collections import Counter
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score as cvs


TMP = 0

#noise = {'the','i','you','and','to','a','for','my','of','in','is','it',
#         'me','your','im','on','we','that','was','they','by','so','he',
#         'as','she','this'}

punctuation = set(string.punctuation)
letters = set(string.ascii_letters)
numbers = set(string.digits)

def remove_hashtag(input_text):
    words = input_text.split()
    for w in words:
        if w[0]=='#':
            w = w[1:]
    return ' '.join(words)

def remove_interpunction(input_text):
    global TMP
    
    if input_text:
        words = input_text.split()
        new_words = []
        for w in words:
            new_words.append(''.join(ch for ch in w if ch not in punctuation))
        return ' '.join(new_words)
    else:
        return input_text

def remove_ats(input_text):
    words = input_text.split()
    for w in words:
        if w.startswith('@'):
            words.remove(w)
    return ' '.join(words) 

def find_most_common(input_text,n=3):
    words = input_text.split()
    word_counter = Counter(words).most_common(n)
    if word_counter[0][1] > 1:
        if n > 1:
            return [word_counter[i][0] for i in range(len(word_counter))]
        else:
            return word_counter[0][0]
    else:
        return 0
    

path = 'test.csv'
df = pd.read_csv(path, encoding = "ISO-8859-1")
df = df[['gender','text']]
df = df[(df['gender'] == 'male') | (df['gender'] == 'female') ]

df['raw_text'] = [(remove_hashtag(x.lower())) for x in df['text']]   
print('Hashtags removed: ') 

df['raw_text'] = [remove_ats(x) for x in df['raw_text']]   
print('@ removed: ')

df['raw_text'] = [remove_interpunction(x) for x in df['raw_text']]   
print('Interpunction removed: ')

df['most_common'] = [find_most_common(x,1) for x in df['raw_text']] 
print('Most common used word in each text found ')

df['text_length'] = [len(x) for x in df['text']]

#average word length
df['avg_word_length'] = [np.average([len(x) for x in a.split()]) for a in df['raw_text']]

#removing tweets with average word length higher or equal to 18
df = df[df['avg_word_length']<18]

#dictionary with most common words occurences
d = dict(zip(list(df['most_common'].value_counts().index),
             list(df['most_common'].value_counts().data)))
    
#changing occurences into frequencies in d
tmp = sum(d.values())
for key in d:
    d[key] = d[key]/tmp
    
df['frequency'] = [d[x] for x in df['most_common']]
print("Most common words found and binarized")  

#checking if interpunction is being used
df['interpunction'] = [bool(set(df.loc[i]['text']) & punctuation) for i in df.index]
print("Interpunction checked")  

#counting words
df['words_number'] = [len(x.split()) for x in df['text']]
print('Words counted\n')  


##Gender prediction

clf_attributes = ['text_length','interpunction','frequency','words_number','avg_word_length']
clf_to_predict = 'gender'

#Label encoding
print("Labels are being encoded ...")
le = LabelEncoder()
data = []
for a in clf_attributes:
    le.fit(df[a])
    data.append(le.transform(df[a]))
    print(a + ' label encoded')

#transpose data matrix
data = list(map(list, zip(*data)))
target = df[clf_to_predict]

print("All labels are encoded\n")  
print("Classification:")
clf =  RandomForestClassifier()
clf.fit(data,target)
feat_imp = clf.feature_importances_
indices = np.argsort(feat_imp)[::-1]
for i in indices:
    print("Feature importance of {a}: {b:4.2f}%".format(a = clf_attributes[i],b = 100*feat_imp[i]))
print()
score = 100*np.average(cvs(clf, data, target, cv=10))
print("Cross validation score is: % 4.2f" % score + "%")
