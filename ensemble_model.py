import sklearn
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import re, string, unicodedata
import nltk
import contractions
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


data = pd.read_csv('/Users/yangren/Desktop/random_tweets.csv')

print(data.shape)
print(data.columns)
data.head(10)
df1 = pd.DataFrame(data = data)

#df1 = df1[df1.language == 'en']
#print(df1)

def replace_contractions(text):
    return contractions.fix(text)

#df1['cleanLinks'] = df1['tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])

emoji_pattern = re.compile("["
                       u"\U0001F600-\U0001F64F"  # emoticons
                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                       u"\U00002702-\U000027B0"
                       u"\U000024C2-\U0001F251"
                       "]+", flags=re.UNICODE)

def remove_emoji(string):
    return emoji_pattern.sub(r'', string)

#df1['cleanemoji'] = df1['cleanLinks'].apply(remove_emoji)

#df1['cleanmention'] = df1['cleanemoji'].replace("@[A-Za-z0-9]+", "",regex=True)
#df1['label'] = '0'
#df1 = df1.drop(['cleanemoji', 'cleanLinks', 'tweet', 'language'], axis=1)

stop_words = stopwords.words("english")
wordnet = WordNetLemmatizer()
def text_preproc(x):
  x = x.lower()
  x = ' '.join([word for word in x.split(' ') if word not in stop_words])
  x = x.encode('ascii', 'ignore').decode()
  x = re.sub(r'#\S+', ' ', x)
  x = re.sub(r'\'\w+', '', x)
  x = re.sub('[%s]' % re.escape(string.punctuation), ' ', x)
  x = re.sub(r'\w*\d+\w*', '', x)
  x = re.sub(r'\s{2,}', ' ', x)
  return x
#df1['clean_tweet'] = df1.cleanmention.apply(text_preproc)
#df1 = df1.drop(['cleanmention'], axis=1)
df1['label'] = '0'
#df1['clean_tweet'] = [entry.lower() for entry in df1['clean_tweet']]
df1['clean_tweet'] = [word_tokenize(entry) for entry in df1['tweet']]
df1 = df1[df1['clean_tweet'].str.len() >= 10]
df1 = df1.drop(['tweet'], axis=1)
print(df1)

data2 = pd.read_csv('/Users/yangren/Desktop/vaping_tweets.csv')
print(data.shape)
print(data.columns)
data.head(10)
df2 = pd.DataFrame(data = data2)

df2['clean_tweet'] = df2.tweet.apply(text_preproc)
df2 = df2.drop(['tweet'], axis=1)
df2['label'] = '1'
df2['clean_tweet'] = [entry.lower() for entry in df2['clean_tweet']]
df2['clean_tweet'] = [word_tokenize(entry) for entry in df2['clean_tweet']]
df2 = df2[df2['clean_tweet'].str.len() >= 10]
print(df2)

df = pd.concat([df1, df2])
print(df)
print(df.columns)
df.to_csv('/Users/yangren/Desktop/cleaned_tw.csv')

seed = 50
df['clean_tweet']=[" ".join(clean_tweet) for clean_tweet in df['clean_tweet'].values]
'''
train = df[df['month'].isin((8, 9))]
test = df[~df['month'].isin((8, 9))]
print(train)
print(test)
Train_X = train['clean_tweet']
print(Train_X)
Train_Y = train['label']
Test_X = test['clean_tweet']
Test_Y = test['label']
'''
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['clean_tweet'],df['label'],test_size=0.5, random_state = seed)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

Tfidf_vect = TfidfVectorizer(max_features=1000)
Tfidf_vect.fit(df['clean_tweet'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

x_test = Test_X_Tfidf
x_train = Train_X_Tfidf
y_train = Train_Y


class ClassifierModel(object):
    def __init__(self, clf, params=None):
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def predict(self, x):
        return self.clf.predict(x)

def trainModel(model, x_train, y_train, x_test, n_folds, seed):
    cv = KFold(n_splits= n_folds, random_state=seed, shuffle=True)
    scores = cross_val_score(model.clf, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    scores1 = cross_val_score(model.clf, x_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
    y_pred = cross_val_predict(model.clf, x_train, y_train, cv=cv, n_jobs=-1)
    return scores, scores1, y_pred

rf_params = {
    'n_estimators': 500,
    'min_samples_split': 8,
    'min_samples_leaf': 1,
    'max_features': 'auto',
    'max_depth': 70,
    'bootstrap': False

}
rfc_model = ClassifierModel(clf=RandomForestClassifier, params=rf_params)
rfc_scores, rfc_scores1, rfc_train_pred = trainModel(rfc_model,x_train, y_train, x_test, 5, 0)
rfc_mean_score = rfc_scores.mean()
rfc_mean_scores1 = rfc_scores1.mean()
print(rfc_scores)
print(rfc_scores1)
print("Mean_RF_Acc : ", rfc_mean_score)
print("Mean_F1_Acc : ", rfc_mean_scores1)
bagging_params = {
    'base_estimator': DecisionTreeClassifier(),
    'n_estimators': 100,
    'random_state':seed
}
'''
bagging_model = ClassifierModel(clf=BaggingClassifier, params=bagging_params)
#bagging_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=seed)
bagging_scores, bagging_scores1, bagging_train_pred = trainModel(bagging_model,x_train, y_train, x_test, 5, 0)
bagging_mean_score = bagging_scores.mean()
bagging_mean_scores1 = bagging_scores1.mean()
print(bagging_scores)
print(bagging_scores1)
print("Mean_bg_Acc : ", bagging_mean_score)
print("Mean_F1_Acc : ", bagging_mean_scores1)
'''
et_params = {
    'n_estimators': 100,
    'max_depth': 5
}
etc_model = ClassifierModel(clf=ExtraTreesClassifier, params=et_params)
etc_scores, etc_scores1, etc_train_pred = trainModel(etc_model,x_train, y_train, x_test, 5, 0)
etc_mean_score = etc_scores.mean()
etc_mean_score1 = etc_scores1.mean()
print(etc_scores)
print(etc_scores1)
print("Mean_ETC_Acc : ", etc_mean_score)
print("Mean_F1_Acc : ", etc_mean_score1)

ada_params = {
    'n_estimators': 100
}
ada_model = ClassifierModel(clf=AdaBoostClassifier, params=ada_params)
ada_scores, score1, ada_train_pred = trainModel(ada_model,x_train, y_train, x_test, 5, 0) # Random Forest
print(ada_scores)
print(score1)
print("Mean_ADA_Acc : ", ada_scores.mean())
print("F1 : ", score1.mean())
gb_params = {
    'n_estimators': 100
}
gbc_model = ClassifierModel(clf=GradientBoostingClassifier, params=gb_params)
gbc_scores, score1, gbc_train_pred = trainModel(gbc_model,x_train, y_train, x_test, 5, 0) # Random Forest
print(gbc_scores)
print("Mean_GBC_Acc : ", gbc_scores.mean())
print("F1 : ", score1.mean())

acc_pred_train = pd.DataFrame ({'RandomForest': rfc_scores.ravel(),
     #'Bagging': bagging_scores.ravel(),
     'ExtraTrees': etc_scores.ravel(),
     'AdaBoost': ada_scores.ravel(),
      'GradientBoost': gbc_scores.ravel()
    })
print(acc_pred_train.head())

x_train = np.column_stack(( etc_train_pred, rfc_train_pred, ada_train_pred, gbc_train_pred))
print(x_train.shape)
print(acc_pred_train)


def trainStackModel(x_train, y_train, x_test, n_folds, seed):
    cv = KFold(n_splits=n_folds, random_state=seed, shuffle=True)
    gbm = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        objective='binary:logistic',
        scale_pos_weight=1).fit(x_train, y_train)
    scores = cross_val_score(gbm, x_train, y_train, scoring='accuracy', cv=cv)
    scores1 = cross_val_score(gbm, x_train, y_train, scoring='roc_auc', cv=cv)
    return scores, scores1
stackModel_scores, scores1 = trainStackModel(x_train, y_train, x_test, 5, 0)
acc_pred_train['stackingModel'] = stackModel_scores
print(acc_pred_train)
print(scores1)
print("F1 : ", scores1.mean())