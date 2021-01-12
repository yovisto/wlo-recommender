# -*- coding: utf-8 -*-
import os, sys, re, nltk, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dot, Reshape, Dense
from tensorflow.keras.models import Model
from sklearn import metrics
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
from nltk.corpus import stopwords
import random
random.seed(100)

dataFile = sys.argv[1]
if not os.path.isfile(dataFile):
    print("File '" + dataFile + "' does not exits.")
    sys.exit(1)

print("Num GPUs available: ", len(tf.config.experimental.list_physical_devices('GPU')))

### LOAD AND PREPROCESS THE DATASET
df = pd.read_csv(dataFile,sep=',')
#df = pd.read_csv("doc2vec.csv",sep=',')
df.columns = ['text', 'id']
#df = df[0:1000]

df = df.reset_index(drop=True)
REPLACE_BY_SPACE_RE = re.compile('[/(){}_\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-zäöüß ]')
STOPWORDS = set(stopwords.words('german')).union(set(stopwords.words('english')))

def clean_text(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

#print (df['text'][:5])
df['text'] = df['text'].apply(clean_text)
df['text'] = df['text'].str.replace('\d+', '')
#print (df['text'][:5])


#### TOKENIZE AND CLEAN TEXT
# The maximum number of words to be used. (most frequent)
MAX_DICT_SIZE = 10000
# Max number of words in each text.
MAX_SEQUENCE_LENGTH = 50

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_DICT_SIZE, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['text'].values)
word_trans = tokenizer.word_index
print('Found %s unique tokens.' % len(word_trans))

X = tokenizer.texts_to_sequences(df['text'].values)

pairs=[]
words=[]

for d in range(len(X)):
    for w in X[d]:
        words.append(w)

words = set(words)  
print ("Num of words: ", len(words))

word_index = {word: idx for idx, word in enumerate(words)}
index_word = {idx: word for word, idx in word_index.items()}
        
for d in range(len(X)):
    for w in X[d]:
        pairs.append((d, word_index[w]))

print ("Num of pairs: ",len(pairs))
pairs_set=set(pairs)


def generate_batch(pairs, n_positive = 50, negative_ratio = 1.0, classification = False):
    """Generate batches of samples for training"""
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))
    
    # Adjust label based on task
    if classification:
        neg_label = 0
    else:
        neg_label = -1
    
    # This creates a generator
    while True:
        # randomly choose positive examples
        for idx, (book_id, link_id) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (book_id, link_id, 1)

        # Increment idx by 1
        idx += 1
        
        # Add negative examples until reach batch size
        while idx < batch_size:
            
            # random selection
            random_book = random.randrange(len(X))
            random_link = random.randrange(len(word_index))
            
            # Check to make sure this is not a positive example
            if (random_book, random_link) not in pairs_set:
                
                # Add to batch and increment index
                batch[idx, :] = (random_book, random_link, neg_label)
                idx += 1
                
        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {'doc': batch[:, 0], 'word': batch[:, 1]}, batch[:, 2]

def embedding_model(embedding_size = 20, classification = False):
    """Model to embed books and wikilinks using the functional API.
       Trained to discern if a link is present in a article"""
    
    # Both inputs are 1-dimensional
    book = Input(name = 'doc', shape = [1])
    link = Input(name = 'word', shape = [1])
    
    # Embedding the book (shape will be (None, 1, 50))
    book_embedding = Embedding(name = 'doc_embedding',
                               input_dim = len(X),
                               output_dim = embedding_size)(book)
    
    # Embedding the link (shape will be (None, 1, 50))
    link_embedding = Embedding(name = 'word_embedding',
                               input_dim = len(word_index),
                               output_dim = embedding_size)(link)
    
    # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
    merged = Dot(name = 'dot_product', normalize = True, axes = 2)([book_embedding, link_embedding])
    
    # Reshape to be a single number (shape will be (None, 1))
    merged = Reshape(target_shape = [1])(merged)

    # If classifcation, add extra layer and loss function is binary cross entropy
    if classification:
        merged = Dense(1, activation = 'sigmoid')(merged)
        model = Model(inputs = [book, link], outputs = merged)
        model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Otherwise loss function is mean squared error
    else:
        model = Model(inputs = [book, link], outputs = merged)
        model.compile(optimizer = 'Adam', loss = 'mse')
    
    return model

#### DEFINE THE MODEL
model = embedding_model()
model.summary()

# PREPARE DATA BATCHES
all = next(generate_batch(pairs, n_positive = len(pairs), negative_ratio = 1))

###### DO THE TRAINING
n_positive = 512
h = model.fit(x=[all[0]['doc'], all[0]['word']], y=all[1], epochs = 15, steps_per_epoch = len(pairs) // n_positive, verbose = 2)

model.save(dataFile.replace(".csv","-embed.h5"))

# store document ids 
with open(dataFile.replace('.csv','-id.pickle'), 'wb') as handle:
    pickle.dump(df["id"], handle, protocol=pickle.HIGHEST_PROTOCOL)

print ("Done :-)")


