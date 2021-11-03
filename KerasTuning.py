import warnings
from kerastuner.tuners import RandomSearch, BayesianOptimization, Hyperband
warnings.filterwarnings('ignore')
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
import re, string, unicodedata
import nltk
import contractions
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
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
df1['label'] = 0

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
df2['label'] = 1

print(df2)

df = pd.concat([df1, df2])
df['clean_tweet'] = [entry.lower() for entry in df['clean_tweet']]
df['clean_tweet_length'] = [word_tokenize(entry) for entry in df['clean_tweet']]
df = df[df['clean_tweet_length'].str.len() >= 10]
df.reset_index(drop=True)
df.to_csv('/Users/yangren/Desktop/Vaping_Keras.csv')
print(df)


from numpy import array
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split
docs = df['clean_tweet']
labels = array(df['label'])

Train_X, Test_X , Train_Y, Test_Y = train_test_split(docs, labels , test_size = 0.50)
vocab_size = 5000
Train_X = [one_hot(d, vocab_size, split=' ') for d in Train_X]
Test_X = [one_hot(d, vocab_size, split=' ') for d in Test_X]
maxlen = 50

x_train = pad_sequences(Train_X, maxlen=maxlen, padding='pre')
x_val = pad_sequences(Test_X, maxlen=maxlen, padding='pre')
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def build_model(hp):
    hp_embed_dim = hp.Int('embed_dim', min_value=32, max_value=512, step = 32)
    hp_ff_dim = hp.Int('ff_dim', min_value=32, max_value=512, step=32)
    hp_num_heads = hp.Choice('num_heads', values=[2, 3])
    hp_dropout = hp.Choice("dropout", values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
    hp_optimizer = hp.Choice('optimizer', values=['adam', 'SGD', 'rmsprop'])
    optimizer = tf.keras.optimizers.get(hp_optimizer)
    optimizer.learning_rate = hp.Choice("learning_rate", [0.1, 0.01, 0.001], default=0.01)
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, hp_embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(hp_embed_dim, hp_num_heads, hp_ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(hp_dropout)(x)
    x = Dense(20, hp_activation)(x)
    x = layers.Dropout(hp_dropout)(x)
    outputs = Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer,
    )
    return model

tuner = RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=15,
    directory=".",
    project_name="keras_trial",
)
tuner.search(x_train, Train_Y, epochs=10, validation_data=(x_val, Test_Y))
best_hps = tuner.get_best_hyperparameters()[0]
print(best_hps.values)

model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, Train_Y, epochs=10, validation_data=(x_val, Test_Y))

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))


