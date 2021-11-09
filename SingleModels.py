import sklearn
import xgboost as xgb
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
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

def replace_contractions(text):
    return contractions.fix(text)

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

df1['cleanLinks'] = df1['tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
df1['cleanemoji'] = df1['cleanLinks'].apply(remove_emoji)
df1['cleanmention'] = df1['cleanemoji'].replace("@[A-Za-z0-9]+", "",regex=True)
df1 = df1.drop(['cleanemoji', 'cleanLinks', 'tweet'], axis=1)

df1['clean_tweet'] = df1.cleanmention.apply(text_preproc)
df1 = df1.drop(['cleanmention'], axis=1)
df1['label'] = '0'
df1['clean_tweet'] = [entry.lower() for entry in df1['clean_tweet']]
df1['clean_tweet'] = [word_tokenize(entry) for entry in df1['clean_tweet']]
df1 = df1[df1['clean_tweet'].str.len() >= 10]
print(df1)

data2 = pd.read_csv('/Users/yangren/Desktop/vaping_tweets.csv')
print(data.shape)
print(data.columns)
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
print(df2)

df = pd.concat([df1, df2])
print(df)
print(df.columns)
df1.to_csv('/Users/yangren/Desktop/cleaned_tw1.csv')
df2.to_csv('/Users/yangren/Desktop/cleaned_tw2.csv')

seed = 50
df['clean_tweet']=[" ".join(clean_tweet) for clean_tweet in df['clean_tweet'].values]
#x = df['clean_tweet']
#y = df['label']

#Encoder = LabelEncoder()
#y = Encoder.fit_transform(y)

Tfidf_vect = TfidfVectorizer(max_features=1000)
Tfidf_vect.fit(df['clean_tweet'])
#x = Tfidf_vect.transform(x)

for holdouts in combinations(df['month'].unique(), 2):
    print(holdouts)
    train = df[df['month'].isin(holdouts)]
    test = df[~df['month'].isin(holdouts)]
    X_train = train['clean_tweet']
    print(X_train)
    y_train = train['label']
    X_test = test['clean_tweet']
    y_test = test['label']

    Encoder = LabelEncoder()
    y_train = Encoder.fit_transform(y_train)
    y_test = Encoder.fit_transform(y_test)

    X_train = Tfidf_vect.transform(X_train)
    X_test = Tfidf_vect.transform(X_test)

    param_grid_nb = {'var_smoothing': np.logspace(0,-9, num=100)}
    nb_grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=5, n_jobs=-1)
    nb_grid.fit(X_train.toarray(), y_train)
    print(nb_grid.best_estimator_)
    #GaussianNB(priors=None, var_smoothing=1.0)
    y_pred = nb_grid.predict(X_test.toarray())
    print(confusion_matrix(y_test, y_pred), ": is the NB confusion matrix")
    print(accuracy_score(y_test, y_pred), ": is the NB accuracy score")
    print(precision_score(y_test, y_pred), ": is the NB precision score")
    print(recall_score(y_test, y_pred), ": is the NB recall score")
    print(f1_score(y_test, y_pred), ": is the NB f1 score")
    
    param_grid_svm = {'kernel': ['rbf', 'poly', 'sigmoid'], 'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001, 0.0001]}
    svm_grid = GridSearchCV(SVC(), param_grid_svm, refit=True, verbose=1, cv=5, n_jobs=1)
    svm_grid.fit(X_train, y_train)
    print(svm_grid.best_estimator_)
    y_pred = svm_grid.predict(X_test)
    print(confusion_matrix(y_test, y_pred), ": is the SVM confusion matrix")
    print(accuracy_score(y_test, y_pred), ": is the SVM accuracy score")
    print(precision_score(y_test, y_pred), ": is the SVM precision score")
    print(recall_score(y_test, y_pred), ": is the SVM recall score")
    print(f1_score(y_test, y_pred), ": is the SVM f1 score")
    
    #randomforest = RandomForestClassifier(
    #bootstrap=True,
    #oob_score=True,
    #random_state=seed,
    #max_features='auto')
    #randomforest.fit(X_train, y_train)
    #test = randomforest.predict(X_test)
    #train = randomforest.predict(X_train)
    #print(classification_report(y_test, test))
    #print("RandomForest Testing Accuracy Score:", accuracy_score(test, y_test) * 100)

    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 100, num=10)]
    max_depth.append(None)
    min_samples_split = [int(x) for x in np.linspace(2, 10, num=5)]
    min_samples_leaf = [int(x) for x in np.linspace(1, 10, num=5)]
    bootstrap = [True, False]

    param_grid_rf = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    rf = RandomForestClassifier()
    rf_grid = RandomizedSearchCV(estimator=rf, param_distributions=param_grid_rf, n_iter=100, cv=5, verbose=1, random_state=50, n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    #print(rf_grid.best_estimator_)
    y_pred = rf_grid.predict(X_test)
    print(rf_grid.best_params_)
    print(confusion_matrix(y_test, y_pred), ": is the RF confusion matrix")
    print(accuracy_score(y_test, y_pred), ": is the RF accuracy score")
    print(precision_score(y_test, y_pred), ": is the RF precision score")
    print(recall_score(y_test, y_pred), ": is the RF recall score")
    print(f1_score(y_test, y_pred), ": is the RF f1 score")
    
    param_xgb = {
        'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'max_depth': np.arange(1, 2),
        'subsample': np.arange(0.5, 1.0, 0.1),
        'colsample_bytree': np.arange(0.4, 1.0, 0.1),
        'eta': np.arange(1, 2, 0.1),
        'gamma': [0, 1, 5]
    }

    xgbr = xgb.XGBRegressor(seed=20)
    clf = RandomizedSearchCV(estimator=xgbr, param_distributions=param_xgb, scoring='neg_mean_squared_error', n_iter=100, cv=5, verbose=1, n_jobs=-1)
    clf.fit(X_train, y_train)
    print("Best parameters:", clf.best_params_)
    print("Lowest RMSE: ", (-clf.best_score_) ** (1 / 2.0))

    y_pred = clf.predict(X_test)
    prediction = [round(value) for value in y_pred]
    print(clf.best_params_)
    test_accuracy = accuracy_score(y_test, prediction)
    print(classification_report(y_test, prediction))
    
    param_mlp = {
        'hidden_layer_sizes': [(100, 50, 100), (100, 100, 100), (500, 250, 500), (100, 100), (500, 500)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'max_iter': np.arange(100, 1000, 100),
        'learning_rate': ['constant', 'adaptive']}
    mlp = MLPClassifier()
    clf = RandomizedSearchCV(estimator=mlp, param_distributions=param_mlp, n_iter=10, cv=5, verbose=1, n_jobs=-1)
    clf.fit(X_train, y_train)
    print("Best parameters:", clf.best_params_)
    y_pred = clf.predict(X_test)
    prediction = [round(value) for value in y_pred]
    print(clf.best_params_)
    test_accuracy = accuracy_score(y_test, prediction)
    print(classification_report(y_test, prediction))
    print("-----------------------------------------------------------------------")


x = df['clean_tweet']
y = df['label']

Encoder = LabelEncoder()
y = Encoder.fit_transform(y)
x = Tfidf_vect.transform(x)

x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.1, random_state=50)
x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.2, random_state=50)
x_train3, x_test3, y_train3, y_test3 = train_test_split(x, y, test_size=0.3, random_state=50)

x_train4, x_test4, y_train4, y_test4 = train_test_split(x, y, test_size=0.4, random_state=50)
x_train5, x_test5, y_train5, y_test5 = train_test_split(x, y, test_size=0.5, random_state=50)

param_grid_nb = {'var_smoothing': np.logspace(0, -9, num=100)}
nb_grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=5, n_jobs=-1)

nb_grid.fit(x_train1.toarray(), y_train1)
print(nb_grid.best_params_, ": is the 10% testing best estimator")
GaussianNB(priors=None, var_smoothing=1.0)
y_pred1 = nb_grid.predict(x_test1.toarray())
print(confusion_matrix(y_test1, y_pred1), ": is the NB 10% testing confusion matrix")
print(accuracy_score(y_test1, y_pred1), ": is the NB 10% testing accuracy score")
print(precision_score(y_test1, y_pred1), ": is the NB 10% testing precision score")
print(recall_score(y_test1, y_pred1), ": is the NB 10% testing recall score")
print(f1_score(y_test1, y_pred1), ": is the NB 10% testing f1 score")

nb_grid.fit(x_train2.toarray(), y_train2)
print(nb_grid.best_params_, ": is the 20% testing best estimator")
GaussianNB(priors=None, var_smoothing=1.0)
y_pred2 = nb_grid.predict(x_test2.toarray())
print(confusion_matrix(y_test2, y_pred2), ": is the NB 20% testing confusion matrix")
print(accuracy_score(y_test2, y_pred2), ": is the NB 20% testing accuracy score")
print(precision_score(y_test2, y_pred2), ": is the NB 20% testing precision score")
print(recall_score(y_test2, y_pred2), ": is the NB 20% testing recall score")
print(f1_score(y_test2, y_pred2), ": is the NB 20% testing f1 score")

nb_grid.fit(x_train3.toarray(), y_train3)
print(nb_grid.best_params_, ": is the 30% testing best estimator")
GaussianNB(priors=None, var_smoothing=1.0)
y_pred3 = nb_grid.predict(x_test3.toarray())
print(confusion_matrix(y_test3, y_pred3), ": is the NB 30% testing confusion matrix")
print(accuracy_score(y_test3, y_pred3), ": is the NB 30% testing accuracy score")
print(precision_score(y_test3, y_pred3), ": is the NB 30% testing precision score")
print(recall_score(y_test3, y_pred3), ": is the NB 30% testing recall score")
print(f1_score(y_test3, y_pred3), ": is the NB 30% testing f1 score")

nb_grid.fit(x_train4.toarray(), y_train4)
print(nb_grid.best_params_, ": is the 40% testing best estimator")
GaussianNB(priors=None, var_smoothing=1.0)
y_pred4 = nb_grid.predict(x_test4.toarray())
print(confusion_matrix(y_test4, y_pred4), ": is the NB 40% testing confusion matrix")
print(accuracy_score(y_test4, y_pred4), ": is the NB 40% testing accuracy score")
print(precision_score(y_test4, y_pred4), ": is the NB 40% testing precision score")
print(recall_score(y_test4, y_pred4), ": is the NB 40% testing recall score")
print(f1_score(y_test4, y_pred4), ": is the NB 40% testing f1 score")

nb_grid.fit(x_train5.toarray(), y_train5)
print(nb_grid.best_params_, ": is the 50% testing best estimator")
GaussianNB(priors=None, var_smoothing=1.0)
y_pred5 = nb_grid.predict(x_test5.toarray())
print(confusion_matrix(y_test5, y_pred5), ": is the NB 50% testing confusion matrix")
print(accuracy_score(y_test5, y_pred5), ": is the NB 50% testing accuracy score")
print(precision_score(y_test5, y_pred5), ": is the NB 50% testing precision score")
print(recall_score(y_test5, y_pred5), ": is the NB 50% testing recall score")
print(f1_score(y_test5, y_pred5), ": is the NB 50% testing f1 score")

param_grid_svm = {'kernel': ['rbf', 'poly', 'sigmoid'], 'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001, 0.0001]}
svm_grid = GridSearchCV(SVC(), param_grid_svm, refit=True, verbose=1, cv=5, n_jobs=1)

svm_grid.fit(x_train1, y_train1)
print(svm_grid.best_estimator_)
y_pred1 = svm_grid.predict(x_test1)
print(confusion_matrix(y_test1, y_pred1), ": is the SVM 10% testing confusion matrix")
print(accuracy_score(y_test1, y_pred1), ": is the SVM 10% testing accuracy score")
print(precision_score(y_test1, y_pred1), ": is the SVM 10% testing precision score")
print(recall_score(y_test1, y_pred1), ": is the SVM 10% testing recall score")
print(f1_score(y_test1, y_pred1), ": is the SVM 10% testing f1 score")

svm_grid.fit(x_train2, y_train2)
print(svm_grid.best_estimator_)
y_pred2 = svm_grid.predict(x_test2)
print(confusion_matrix(y_test2, y_pred2), ": is the SVM 20% testing confusion matrix")
print(accuracy_score(y_test2, y_pred2), ": is the SVM 20% testing accuracy score")
print(precision_score(y_test2, y_pred2), ": is the SVM 20% testing precision score")
print(recall_score(y_test2, y_pred2), ": is the SVM 20% testing recall score")
print(f1_score(y_test2, y_pred2), ": is the SVM 20% testing f1 score")

svm_grid.fit(x_train3, y_train3)
print(svm_grid.best_estimator_)
y_pred3 = svm_grid.predict(x_test3)
print(confusion_matrix(y_test3, y_pred3), ": is the 30% testing SVM confusion matrix")
print(accuracy_score(y_test3, y_pred3), ": is the 30% testing SVM accuracy score")
print(precision_score(y_test3, y_pred3), ": is the 30% testing SVM precision score")
print(recall_score(y_test3, y_pred3), ": is the 30% testing SVM recall score")
print(f1_score(y_test3, y_pred3), ": is the 30% testing SVM f1 score")

svm_grid.fit(x_train4, y_train4)
print(svm_grid.best_estimator_)
y_pred4 = svm_grid.predict(x_test4)
print(confusion_matrix(y_test4, y_pred4), ": is the 40% testing SVM confusion matrix")
print(accuracy_score(y_test4, y_pred4), ": is the 40% testing SVM accuracy score")
print(precision_score(y_test4, y_pred4), ": is the 40% testing SVM precision score")
print(recall_score(y_test4, y_pred4), ": is the 40% testing SVM recall score")
print(f1_score(y_test4, y_pred4), ": is the 40% testing SVM f1 score")

svm_grid.fit(x_train5, y_train5)
print(svm_grid.best_estimator_)
y_pred5 = svm_grid.predict(x_test5)
print(confusion_matrix(y_test5, y_pred5), ": is the 50% testing SVM confusion matrix")
print(accuracy_score(y_test5, y_pred5), ": is the 50% testing SVM accuracy score")
print(precision_score(y_test5, y_pred5), ": is the 50% testing SVM precision score")
print(recall_score(y_test5, y_pred5), ": is the 50% testing SVM recall score")
print(f1_score(y_test5, y_pred5), ": is the 50% testing SVM f1 score")

n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 100, num=10)]
max_depth.append(None)
min_samples_split = [int(x) for x in np.linspace(2, 10, num=5)]
min_samples_leaf = [int(x) for x in np.linspace(1, 10, num=5)]
bootstrap = [True, False]

param_grid_rf = {'n_estimators': n_estimators,
                 'max_features': max_features,
                 'max_depth': max_depth,
                 'min_samples_split': min_samples_split,
                 'min_samples_leaf': min_samples_leaf,
                 'bootstrap': bootstrap}
rf = RandomForestClassifier()
rf_grid = RandomizedSearchCV(estimator=rf, param_distributions=param_grid_rf, n_iter=100, cv=5, verbose=1,
                             random_state=50, n_jobs=-1)

rf_grid.fit(x_train1, y_train1)
# print(rf_grid.best_estimator_)
y_pred1 = rf_grid.predict(x_test1)
print(rf_grid.best_params_)
print(confusion_matrix(y_test1, y_pred1), ": is the 10% testing RF confusion matrix")
print(accuracy_score(y_test1, y_pred1), ": is the 10% testing RF accuracy score")
print(precision_score(y_test1, y_pred1), ": is the 10% testing RF precision score")
print(recall_score(y_test1, y_pred1), ": is the 10% testing RF recall score")
print(f1_score(y_test1, y_pred1), ": is the 10% testing RF f1 score")

rf_grid.fit(x_train2, y_train2)
# print(rf_grid.best_estimator_)
y_pred2 = rf_grid.predict(x_test2)
print(rf_grid.best_params_)
print(confusion_matrix(y_test2, y_pred2), ": is the 20% testing RF confusion matrix")
print(accuracy_score(y_test2, y_pred2), ": is the 20% testing RF accuracy score")
print(precision_score(y_test2, y_pred2), ": is the 20% testing RF precision score")
print(recall_score(y_test2, y_pred2), ": is the 20% testing RF recall score")
print(f1_score(y_test2, y_pred2), ": is the 20% testing RF f1 score")

rf_grid.fit(x_train3, y_train3)
# print(rf_grid.best_estimator_)
y_pred3 = rf_grid.predict(x_test3)
print(rf_grid.best_params_)
print(confusion_matrix(y_test3, y_pred3), ": is the 30% testing RF confusion matrix")
print(accuracy_score(y_test3, y_pred3), ": is the 30% testing RF accuracy score")
print(precision_score(y_test3, y_pred3), ": is the 30% testing RF precision score")
print(recall_score(y_test3, y_pred3), ": is the 30% testing RF recall score")
print(f1_score(y_test3, y_pred3), ": is the 30% testing RF f1 score")

rf_grid.fit(x_train4, y_train4)
# print(rf_grid.best_estimator_)
y_pred4 = rf_grid.predict(x_test4)
print(rf_grid.best_params_)
print(confusion_matrix(y_test4, y_pred4), ": is the 40% testing RF confusion matrix")
print(accuracy_score(y_test4, y_pred4), ": is the 40% testing RF accuracy score")
print(precision_score(y_test4, y_pred4), ": is the 40% testing RF precision score")
print(recall_score(y_test4, y_pred4), ": is the 40% testing RF recall score")
print(f1_score(y_test4, y_pred4), ": is the 40% testing RF f1 score")

rf_grid.fit(x_train5, y_train5)
# print(rf_grid.best_estimator_)
y_pred5 = rf_grid.predict(x_test5)
print(rf_grid.best_params_)
print(confusion_matrix(y_test5, y_pred5), ": is the 50% testing RF confusion matrix")
print(accuracy_score(y_test5, y_pred5), ": is the 50% testing RF accuracy score")
print(precision_score(y_test5, y_pred5), ": is the 50% testing RF precision score")
print(recall_score(y_test5, y_pred5), ": is the 50% testing RF recall score")
print(f1_score(y_test5, y_pred5), ": is the 50% testing RF f1 score")

param_xgb = {
    'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'max_depth': np.arange(1, 2),
    'subsample': np.arange(0.5, 1.0, 0.1),
    'colsample_bytree': np.arange(0.4, 1.0, 0.1),
    'eta': np.arange(1, 2, 0.1),
    'gamma': [0, 1, 5]
}

xgbr = xgb.XGBRegressor(seed=20)
clf = RandomizedSearchCV(estimator=xgbr, param_distributions=param_xgb, scoring='neg_mean_squared_error', n_iter=100,
                         cv=5, verbose=1, n_jobs=-1)

clf.fit(x_train1, y_train1)
print("Best parameters:", clf.best_params_)
print("Lowest RMSE: ", (-clf.best_score_) ** (1 / 2.0))

y_pred1 = clf.predict(x_test1)
prediction1 = [round(value) for value in y_pred1]
print(clf.best_params_)
test_accuracy = accuracy_score(y_test1, prediction1)
print(classification_report(y_test1, prediction1))

clf.fit(x_train2, y_train2)
print("Best parameters:", clf.best_params_)
print("Lowest RMSE: ", (-clf.best_score_) ** (1 / 2.0))

y_pred2 = clf.predict(x_test2)
prediction2 = [round(value) for value in y_pred2]
print(clf.best_params_)
test_accuracy = accuracy_score(y_test2, prediction2)
print(classification_report(y_test2, prediction2))

clf.fit(x_train3, y_train3)
print("Best parameters:", clf.best_params_)
print("Lowest RMSE: ", (-clf.best_score_) ** (1 / 2.0))

y_pred3 = clf.predict(x_test3)
prediction3 = [round(value) for value in y_pred3]
print(clf.best_params_)
test_accuracy = accuracy_score(y_test3, prediction3)
print(classification_report(y_test3, prediction3))


clf.fit(x_train4, y_train4)
print("Best parameters:", clf.best_params_)
print("Lowest RMSE: ", (-clf.best_score_) ** (1 / 2.0))

y_pred4 = clf.predict(x_test4)
prediction4 = [round(value) for value in y_pred4]
print(clf.best_params_)
test_accuracy = accuracy_score(y_test4, prediction4)
print(classification_report(y_test4, prediction4))


clf.fit(x_train5, y_train5)
print("Best parameters:", clf.best_params_)
print("Lowest RMSE: ", (-clf.best_score_) ** (1 / 2.0))

y_pred5 = clf.predict(x_test5)
prediction5 = [round(value) for value in y_pred5]
print(clf.best_params_)
test_accuracy = accuracy_score(y_test5, prediction5)
print(classification_report(y_test5, prediction5))


param_mlp = {
    'hidden_layer_sizes': [(100, 50, 100), (100, 100, 100), (500, 250, 500), (100, 100), (500, 500)],
    'activation': ['tanh', 'relu', 'logistic'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'max_iter': np.arange(100, 1000, 100),
    'learning_rate': ['constant', 'adaptive']}
mlp = MLPClassifier()
clf = RandomizedSearchCV(estimator=mlp, param_distributions=param_mlp, n_iter=10, cv=5, verbose=1, n_jobs=-1)

clf.fit(x_train1, y_train1)
print("Best parameters:", clf.best_params_)
y_pred1 = clf.predict(x_test1)
prediction1 = [round(value) for value in y_pred1]
print(clf.best_params_)
test_accuracy = accuracy_score(y_test1, prediction1)
print(classification_report(y_test1, prediction1))

clf.fit(x_train2, y_train2)
print("Best parameters:", clf.best_params_)
y_pred2 = clf.predict(x_test2)
prediction2 = [round(value) for value in y_pred2]
print(clf.best_params_)
test_accuracy = accuracy_score(y_test2, prediction2)
print(classification_report(y_test2, prediction2))

clf.fit(x_train3, y_train3)
print("Best parameters:", clf.best_params_)
y_pred3 = clf.predict(x_test3)
prediction3 = [round(value) for value in y_pred3]
print(clf.best_params_)
test_accuracy = accuracy_score(y_test3, prediction3)
print(classification_report(y_test3, prediction3))

clf.fit(x_train4, y_train4)
print("Best parameters:", clf.best_params_)
y_pred4 = clf.predict(x_test4)
prediction4 = [round(value) for value in y_pred4]
print(clf.best_params_)
test_accuracy = accuracy_score(y_test4, prediction4)
print(classification_report(y_test4, prediction4))

clf.fit(x_train5, y_train5)
print("Best parameters:", clf.best_params_)
y_pred5 = clf.predict(x_test5)
prediction5 = [round(value) for value in y_pred5]
print(clf.best_params_)
test_accuracy = accuracy_score(y_test5, prediction5)
print(classification_report(y_test5, prediction5))
