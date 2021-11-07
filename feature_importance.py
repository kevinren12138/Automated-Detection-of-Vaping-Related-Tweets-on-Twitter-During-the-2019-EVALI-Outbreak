import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import datasets
import pandas
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier

from sklearn.model_selection import cross_val_score, train_test_split

from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions
from sklearn.ensemble import VotingClassifier
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
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
'''
data = pd.read_csv('/Users/yangren/Desktop/random_tweets.csv')
print(data.shape)
print(data.columns)
data.head(10)
df1 = pd.DataFrame(data = data)

def replace_contractions(text):
    return contractions.fix(text)

df1['cleanLinks'] = df1['tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])

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

df1['cleanemoji'] = df1['cleanLinks'].apply(remove_emoji)

df1['cleanmention'] = df1['cleanemoji'].replace("@[A-Za-z0-9]+", "",regex=True)

#df1['label'] = '0'
df1 = df1.drop(['cleanemoji', 'cleanLinks', 'tweet'], axis=1)

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
df1['clean_tweet'] = df1.cleanmention.apply(text_preproc)
df1 = df1.drop(['cleanmention'], axis=1)
df1['label'] = '0'
df1['clean_tweet'] = [entry.lower() for entry in df1['clean_tweet']]
df1['clean_tweet'] = [word_tokenize(entry) for entry in df1['clean_tweet']]
df1 = df1[df1['clean_tweet'].str.len() >= 10]

data2 = pd.read_csv('/Users/yangren/Desktop/vaping_tweets.csv')
data.head(10)
df2 = pd.DataFrame(data = data2)
df2['cleanLinks'] = df2['tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
df2['cleanemoji'] = df2['cleanLinks'].apply(remove_emoji)
df2['cleanmention'] = df2['cleanemoji'].replace("@[A-Za-z0-9]+", "",regex=True)
df2 = df2.drop(['cleanemoji', 'cleanLinks', 'tweet'], axis=1)

df2['clean_tweet'] = df2.cleanmention.apply(text_preproc)
df2 = df2.drop(['cleanmention'], axis=1)
df2['label'] = '1'
df2['clean_tweet'] = [entry.lower() for entry in df2['clean_tweet']]
df2['clean_tweet'] = [word_tokenize(entry) for entry in df2['clean_tweet']]
df2 = df2[df2['clean_tweet'].str.len() >= 10]

df = pd.concat([df1, df2])
df['clean_tweet']=[" ".join(clean_tweet) for clean_tweet in df['clean_tweet'].values]
df.to_csv('/Users/yangren/Desktop/cleaned_tw.csv')
'''
seed = 50 
data = pd.read_csv('/Users/yangren/Desktop/cleaned_tw.csv')
df = pd.DataFrame(data = data)

x = df['clean_tweet']
y = df['label']

Encoder = LabelEncoder()
y = Encoder.fit_transform(y)
#print(x)
#print(y)

Tfidf_vect = TfidfVectorizer(max_features=1000)
Tfidf_vect.fit(df['clean_tweet'])
x = Tfidf_vect.transform(x)
print(x)
feature_names = Tfidf_vect.get_feature_names()
print(feature_names)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)
randomforest = RandomForestClassifier(
                      bootstrap=True,
                      oob_score=True,
                      random_state=seed,
                      max_features='auto')

randomforest.fit(x_train,y_train)
test = randomforest.predict(x_test)
train = randomforest.predict(x_train)
print(classification_report(y_test, test))
#print("RandomForest Training Accuracy Score:",accuracy_score(train, y_train)*100)
print("RandomForest Testing Accuracy Score:",accuracy_score(test, y_test)*100)
importance = randomforest.feature_importances_
important_features_dict = {}
for x,i in zip(feature_names,importance):
    important_features_dict[x]=i
with plt.style.context('seaborn'):
    plt.figure(figsize=(10,5))
    plt.bar(list(important_features_dict.keys()),list(important_features_dict.values()))
    plt.xticks(rotation=90)
    plt.ylabel('Importance')
    plt.title('Random Forest Classifier Feature Importance')
    plt.show()

indices = np.argsort(importance)

# customized number
num_features = 20

plt.figure(figsize=(10,5))
plt.title('Feature Importances')

# only plot the customized number of features
plt.barh(range(num_features), importance[indices[-num_features:]], color='b', align='center')
plt.yticks(range(num_features), [feature_names[i] for i in indices[-num_features:]])
plt.xticks(rotation=90)
plt.xlabel('Importance')
plt.show()
