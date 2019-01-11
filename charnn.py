#!/usr/bin/env python
import os
import sys
import random
import tflearn
import pickle
import re
from tflearn.data_utils import *

maxlen = 100
batch_size = maxlen * 500
n_epoch = 10
decay = 1
min_epoch = 3

def preprocess(string):
    string = string.lower()
    string = re.sub(r"[^a-яa-zё\\\,\!\?\.\n]", " ", string)
    string = re.sub(r"[\s]+\.", ".", string)
    string = re.sub(r"[\s]+,", ",", string)
    string = re.sub(r"[\s]+!", "!", string)
    string = re.sub(r"[\s]+\?", "?", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def to_semi_redundant_sequences(string, char_idx, seq_maxlen=25, redun_step=3):

    print("Vectorizing text...")

    sequences = []
    next_chars = []
    for i in range(0, len(string) - seq_maxlen, redun_step):
        sequences.append(string[i: i + seq_maxlen])
        next_chars.append(string[i + seq_maxlen])

    X = np.zeros((len(sequences), seq_maxlen, len(char_idx)), dtype=np.bool)
    Y = np.zeros((len(sequences), len(char_idx)), dtype=np.bool)
    for i, seq in enumerate(sequences):
        for t, char in enumerate(seq):
            X[i, t, char_idx[char]] = 1
        Y[i, char_idx[next_chars[i]]] = 1

    print("Text total length: " + str(len(string)))
    print("Distinct chars: " + str(len(char_idx)))
    print("Total sequences: " + str(len(sequences)))

    return X, Y


path = sys.argv[1]

with open(path, 'rt') as f:
    text = preprocess(f.read())

char_idx = {}

path='char_idx.pickle'
if os.path.isfile(path):
    with open(path, 'rb') as f:
        char_idx = pickle.load(f)
else:
    idx = 0
    for b in range(int(len(text)/batch_size)+1):
        batch = text[b*batch_size:min(len(text),(b+1)*batch_size)]
        for c in batch:
            if c not in char_idx.keys():
                char_idx[c] = idx
                idx += 1

    with open('char_idx.pickle', 'wb') as f:
        pickle.dump(char_idx, f)


g = tflearn.input_data([None, maxlen, len(char_idx)])
g = tflearn.lstm(g, 256, return_seq=True)
g = tflearn.dropout(g, 0.9)
g = tflearn.lstm(g, 256, return_seq=True)
g = tflearn.dropout(g, 0.9)
g = tflearn.lstm(g, 256)
g = tflearn.dropout(g, 0.9)
g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                       learning_rate=0.01)

m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                              seq_maxlen=maxlen,
                              clip_gradients=5.0,
                              checkpoint_path='model_char_lstm')

if os.path.isfile('char_lstm.model'):
    m.load('char_lstm.model')

if os.path.isfile('char_lstm_state.pickle'):
    with open('char_lstm_state.pickle','rb') as f:
        (i, b, n_epoch) = pickle.load(f)
else:
    i = b = 0


for i in range(i,10):
    for b in range(b,int(len(text)/batch_size)+1):
        with open('char_lstm_state.pickle', 'wb') as f:
            pickle.dump((i,b,n_epoch),f)
        batch = text[b*batch_size:min(len(text),(b+1)*batch_size)]
        #print(batch)
        print('i: {} b: {}'.format(i,b))
        X, Y = to_semi_redundant_sequences(batch,
                                           char_idx,
                                           seq_maxlen=maxlen,
                                           redun_step=3)

        m.fit(X, Y,
              batch_size=128,
              n_epoch=n_epoch+1,
              validation_set=0.1,
              run_id='model_char_lstm',
              snapshot_epoch=False)
        seed_pos = random.randint(0,len(text)-maxlen)
        seed = text[seed_pos:seed_pos+maxlen]
        for t in (1.0, 0.8, 0.5, 0.1):
            print("'{}'\n".format(m.generate(int(len(seed) + random.random()*200),
                  temperature=t,
                  seq_seed=seed)))
        m.save('char_lstm.model')
        if n_epoch > min_epoch:
            n_epoch = int(n_epoch * decay)

