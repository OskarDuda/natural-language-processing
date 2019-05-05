import category_encoders as ce
import string
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score as cvs, train_test_split
import xgboost

TMP = 0

#noise = {'the','i','you','and','to','a','for','my','of','in','is','it',
#         'me','your','im','on','we','that','was','they','by','so','he',
#         'as','she','this'}

PUNCTUATION = set(string.punctuation)
LETTERS = set(string.ascii_letters)
NUMBERS = set(string.digits)

def remove_hashtag(input_text):
    words = input_text.split()
    for w in words:
        if w[0]=='#':
            w = w[1:]
    return ' '.join(words)

def remove_punctuation(input_text):
    global TMP
    
    if input_text:
        words = input_text.split()
        new_words = []
        for w in words:
            new_words.append(''.join(ch for ch in w if ch not in PUNCTUATION))
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


def load_data(path):
    df = pd.read_csv(path, encoding="ISO-8859-1")
    df = df[['gender','text']]
    df = df[(df['gender'] == 'male') | (df['gender'] == 'female') ]
    return df


def extract_features(data):
    data = data.copy()
    data['raw_text'] = [(remove_hashtag(x.lower())) for x in data['text']]
    print('Hashtags removed: ')

    data['raw_text'] = [remove_ats(x) for x in data['raw_text']]
    print('@ removed: ')

    data['raw_text'] = [remove_punctuation(x) for x in data['raw_text']]
    print('Punctuation removed: ')

    data['most_common'] = [find_most_common(x, 1) for x in data['raw_text']]
    print('Most common used word in each text found ')

    data['text_length'] = [len(x) for x in data['text']]

    data['avg_word_length'] = [np.average([len(x) for x in a.split()]) for a in data['raw_text']]

    data = data[data['avg_word_length'] < 18]  # more than 99% of english words shorter than 18 letters

    #dictionary with most common words occurences
    d = dict(zip(list(data['most_common'].value_counts().index),
                 list(data['most_common'].value_counts().data)))

    #changing occurences into frequencies in d
    tmp = sum(d.values())
    for key in d:
        d[key] = d[key]/tmp

    data['frequency'] = [d[x] for x in data['most_common']]
    print("Most common words found and binarized")

    #checking if punctuation is being used
    data['punctuation'] = [bool(set(data.loc[i]['text']) & PUNCTUATION) for i in data.index]
    print("Punctuation checked")

    #counting words
    data['words_number'] = [len(x.split()) for x in data['text']]
    print('Words counted\n')

    return data


def encode_labels(data, X_columns):
    data = data[X_columns].copy()
    print("Labels are being encoded ...")
    le = ce.OrdinalEncoder()
    data = le.fit_transform(data)
    return data


def build_model(X, y):
    clf = xgboost.XGBClassifier(n_estimators=200)
    clf.fit(X, y)
    return clf


def evaluate_model(clf, X, y):
    feat_imp = dict(zip(X.columns, clf.feature_importances_))
    for feature in feat_imp.keys():
        print("Feature importance of {}: {:4.2f}%".format(feature, 100*feat_imp[feature]))
    score = roc_auc_score(y, clf.predict(X))
    print("\nAUC score is: % 4.2f" % score + "%")
    return score, feat_imp


def main():
    path = 'Data/test.csv'
    df = load_data(path)
    df = extract_features(df)
    df['is_male'] = (df['gender'] == 'male').astype(int)
    X_features = ['most_common', 'text_length', 'punctuation', 'frequency', 'words_number', 'avg_word_length']
    y_features = 'is_male'
    df[X_features] = encode_labels(df, X_features)
    X_train, X_test, y_train, y_test = train_test_split(df[X_features], df[y_features])
    clf = build_model(X_train, y_train)
    score, feature_importance = evaluate_model(clf, X_test, y_test)


if __name__ == '__main__':
    main()
