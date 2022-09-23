#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer

from sklearn import model_selection, metrics, preprocessing, ensemble, model_selection, metrics
from sklearn.feature_extraction.text import CountVectorizer

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Dropout, Input, Embedding
from tensorflow.keras.optimizers import Adam


# In[2]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))


# # Data Reading and Info

# In[3]:


df = pd.read_csv('./train.csv')
df.head()


# In[4]:


df.info()


# # Missing Data

# In[5]:


num_rows, num_features = df.shape
df.apply(lambda x: [x.isna().sum(), x.isna().sum()/num_rows * 100], axis=0).set_index(pd.Series(["# of missing data", "Percentage"]))


# In[6]:


sns.displot(data=df.isna().melt(value_name="missing"), y="variable", hue="missing", multiple="fill", aspect=1) 
plt.show()


# # How many data in each class?

# In[7]:


df["target"].value_counts()


# In[8]:


sns.countplot(x="target", data=df).set(title="Target Count")
plt.show()


# # Top 15 Locations

# In[9]:


locations = df["location"].value_counts()
locations[0:15]


# # Top 15 Keywords

# In[10]:


keywords = df["keyword"].value_counts()
keywords[0:15]


# # Most Common Words

# In[13]:


import collections
from collections import Counter

word_count = Counter(" ".join(df["text"]).split())
words = word_count.most_common(10)
words


# In[15]:


stop_words = stopwords.words('english')
stop = {word: word_count[word] for word in word_count if word in stop_words}
common_stop_words = sorted(stop.items(), key=lambda x: x[1], reverse=True)
common_stop_words[:10]


# # Prepare Data Set

# Concatenate Keyword & Text
# 

# In[16]:


def concat(row):
    if type(row) != list:
        row = row.to_list()
    x, y = row[0], row[1]
    new_cell = x + " " + y if isinstance(x, str) else y
    return new_cell


# In[18]:


df["Key_Text"] = df.loc[:, ["keyword", "text"]].apply(concat, axis=1)
df


# Remove Stop Words

# In[19]:


def remove_stop_words(text):
    tokens = [token.lower() for token in text.split() if token.lower() not in stop_words]
    return " ".join(tokens)


# Remove Numbers

# In[20]:


def remove_number(text):
    return ''.join([ele for ele in text if not ele.isdigit()])


# Stemming

# In[21]:


from nltk.stem import PorterStemmer

ps = PorterStemmer()
def stem(text):
    return ''.join([ps.stem(ele) for ele in text])


# Remove Punctuation

# In[22]:


import string

def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


# Final Preperation

# In[23]:


def PrepareData(text):
    text = remove_stop_words(text)
    text = remove_number(text)
    text = stem(text)
    text = remove_punctuation(text)
    return text


# In[25]:


df["NLP"] = df["Key_Text"].apply(PrepareData)
df


# # Tokenization

# In[26]:


import nltk
nltk.download('punkt')
df["NLP"] = df["NLP"].apply(lambda text: word_tokenize(text))
df


# In[27]:


from sklearn.model_selection import train_test_split

X = df["NLP"]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=44)

print("X_train shape= ", X_train.shape)
print("y_train shape= ", y_train.shape)

print("X_test shape= ", X_test.shape)
print("y_test shape= ", y_test.shape)


# Vocabulary size

# In[28]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
print("Vocabulary Size: ", vocab_size)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)


# # Padding

# In[29]:


seq_len = 50

X_train_pad = pad_sequences(X_train_seq, maxlen=seq_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=seq_len)
X_train_pad


# # Model Structure

# In[30]:


embedding_dim = 32

mdl = Sequential()

mdl.add(Embedding(vocab_size, embedding_dim, input_length=seq_len))
mdl.add(LSTM(64, dropout=0.1))
mdl.add(Dense(1, activation="sigmoid"))

mdl.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])
mdl.summary()


# Train

# In[31]:


mdl.fit(X_train_pad, y_train, epochs=20, validation_data=(X_test_pad, y_test))


# Test

# In[32]:


df_test = pd.read_csv('./test.csv')
df_test.head()


# In[33]:


df_test["Key_Text"] = df_test.loc[:, ["keyword", "text"]].apply(concat, axis=1)
df_test["NLP"] = df_test["Key_Text"].apply(PrepareData)
df_test["NLP"] = df_test["NLP"].apply(lambda text: word_tokenize(text))
df_test


# In[34]:


X_test_test = df_test["NLP"]
X_test_test
X_test_test_seq = tokenizer.texts_to_sequences(X_test_test)
X_test_test_pad = pad_sequences(X_test_test_seq, maxlen=seq_len)


# In[35]:


predicted = mdl.predict(X_test_test_pad)
predicted


# In[36]:


df_test.drop(["NLP", "Key_Text"], axis=1, inplace=True)
df_test["class"] = predicted
df_test["class"] = df_test["class"].apply(lambda x: 1 if x > 0.5 else 0)
df_test.head()


# # Evaluate the model

# In[37]:


predicted = mdl.predict(X_test_pad)

y_predicted = [1 if ele > 0.5 else 0 for ele in predicted]

score, test_accuracy = mdl.evaluate(X_test_pad, y_test)

print("Test Accuracy: ", test_accuracy)
print(metrics.classification_report(list(y_test), y_predicted))


# In[38]:


conf_matrix = metrics.confusion_matrix(y_test, y_predicted)

fig, ax = plt.subplots()
sns.heatmap(conf_matrix, cbar=False, cmap='Reds', annot=True, fmt='d')
ax.set(xlabel="Predicted Value", ylabel="True Value", title="Confusion Matrix")
ax.set_yticklabels(labels=['0', '1'], rotation=0)
plt.show()


# In[39]:


mdl.save("./")


# In[ ]:




