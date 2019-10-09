# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re 
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional, Concatenate
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras import backend as K
from sklearn.metrics import f1_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence

from keras.layers import Concatenate, Permute, Dot, Input, Multiply
from keras.layers import RepeatVector, Lambda
from keras.activations import softmax

from keras import backend as K
from keras.layers import Layer

from keras import initializers as initializers

# Load DATA
EMBEDDING_FILE = "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"
TRAIN_DATA_FILE = "../input/train.csv"
TEST_DATA_FILE = "../input/test.csv"

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")



########################################
## GLOBAL PARAMETERS
########################################

# After studying the training set we define a maximum length of words for sentences.
# We pad everything to 0 for smaller sentences and truncate all slonger sentences.

max_Len_sentence = 70
embed_size=300

# Define a maximum value for features to use from embedding matrix (reduce bias and computaion time)
max_features_embedding =50000

# Manually do some text cleaning
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
'·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
'“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
'▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
'∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
    
def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x
    
def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x
    
mispell_dict = {"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"}
    
def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)

    # Clean the text
train["question_text"] = train["question_text"].apply(lambda x: clean_text(x.lower()))
test["question_text"] = test["question_text"].apply(lambda x: clean_text(x.lower()))
    
    # Clean numbers
train["question_text"] = train["question_text"].apply(lambda x: clean_numbers(x))
test["question_text"] = test["question_text"].apply(lambda x: clean_numbers(x))
    
    # Clean speelings
train["question_text"] = train["question_text"].apply(lambda x: replace_typical_misspell(x))
test["question_text"] = test["question_text"].apply(lambda x: replace_typical_misspell(x))
    


########################################
## TOKKENIZE THE DATAS
########################################

# Tokkenize the data and preprocess 

print('create Tokenizer ')

tokenizer = Tokenizer(num_words=max_features_embedding)
Full_Tokenize_dat = list(train['question_text'].values) + list(test['question_text'].values)
tokenizer.fit_on_texts(Full_Tokenize_dat)

print('Tokenize Datas ')

train_tokenized = tokenizer.texts_to_sequences(train['question_text'].fillna('##_'))
test_tokenized = tokenizer.texts_to_sequences(test['question_text'].fillna('##_'))                                          

#Word Index est retourné par ce qui est entrainé par le Tokenizer
word_index=tokenizer.word_index                                              
                                              
print('Padding Sequences')
                                             
X_train_pad=pad_sequences(train_tokenized,maxlen=max_Len_sentence,padding='post')
X_test_pad=pad_sequences(test_tokenized,maxlen=max_Len_sentence,padding='post')                  

########################################
## EMBEDDING MATRIX
########################################

print('Define and Load Embedding Matrix')

# Manually load GloVe Matrix (as required by Kaggle)
# The idea is to define a maximum number of GloVe features to be taken into consideration

def Load_GloVes(glove_file,word_index):
    
    with open(glove_file,'r',encoding='utf8') as f:
        words= set()
        Glove_Index={}
        
        for lines in f:
            tmpLine = lines.split(" ")
            curr_word=tmpLine[0]
            words.add(curr_word)
            Glove_Index[curr_word] = np.array(tmpLine[1:], dtype='float32')
    
    # We define mean and std of embedding matrix to built the embedding matrix with a maximum size
    words=list(Glove_Index.values())
    emb_mean,emb_std = np.array(words).mean(), np.array(words).std()
    
    # Set a maximum size to the embedding matrix given the Training set
    nb_words_emb=min(max_features_embedding,len(word_index))
    
    # Initiallize the matrix with means and std.
    # Add +1 to the shape to fit KERAS requirements
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words_emb+1, embed_size))
     
    for word, i in word_index.items():
        if i >= max_features_embedding: continue
        embedding_vector = Glove_Index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        
    return embedding_matrix



# Load Dict and Embeddings

embedding_matrix=Load_GloVes(EMBEDDING_FILE,word_index)                            
                                              


########################################
## LTSM MODEL : DEFINITION
########################################

print('Define LSTM Model')

# User define Softmax to deal with axis problems
def softMaxAxis1(x):
    return softmax(x,axis=1)

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class Attention_HCN (Layer):

    # Idea is to use Attention to define weights
    
    ATTENTION Network
    
    def __init__(self, bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.bias = bias
        super(Attention_HCN, self).__init__(**kwargs)


    def build(self, input_shape):
        
        # this is where you will define your weights. This method must set self.built = True at the end, which can be done by calling super([Layer], self).build()
        # on défini les variables sur lequelles on va apprendre
        
        assert len(input_shape) == 3
        
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),initializer=self.init,name='{}_W'.format(self.name))

        if self.bias:
            self.b = self.add_weight((input_shape[-1],),initializer='zero',name='{}_b'.format(self.name))

        self.u_word_level = self.add_weight((input_shape[-1],),initializer=self.init,name='{}_u'.format(self.name))
                                 
        super(Attention_HCN, self).build(input_shape)                         
 

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
                                
                        
    def call(self,x,mask=None):

        # define ui en fonction de hi
        uit = dot_product(x,self.W)
        uit += self.b
        uit = K.tanh(uit)
        
        # define aplha i en fonction du word context vector sur lequel on apprend
        ait = dot_product(uit, self.u_word_level)
        
        a = K.exp(ait)
        
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        
        #calculate acti
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        
        # weighed context vector
        weighted_input = x * a

        return K.sum(weighted_input, axis=1)


    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]



def Attention_Architecture(input_shape,embedding_matrix):
    # Define the Model architecture
    # Concatenate results of the usual LSTM and 
    
    emb_size=embedding_matrix.shape[1]
    max_features_embedding=embedding_matrix.shape[0]

    # Input Shape est de la dimnension (m,Tx)
    Activation_0 = Input(shape=input_shape)

    emb = Embedding(max_features_embedding,emb_size, weights=[embedding_matrix],trainable=False)(Activation_0)
    
    X = Bidirectional(LSTM(128,return_sequences=True),merge_mode='concat')(emb)
    
    #X = Dropout(0.3)(X)  
    
    context = Attention_HCN()(X)
    lstm = LSTM(32,return_sequences=False)(X)     
      
    conc = Concatenate()([context,lstm])         

    X = Dense(8,activation='relu')(conc)  
    #X = Dropout(0.3)(X) 
    #X = Dropout(0.5)(X)
    # Dense = Sigmoid // One output then we chose the Threshold equal to 0/5
    X_out = Dense(1,activation="sigmoid",name='output_sigmoid')(X)
    #@X_out=Flatten()(X_out)
    # Define the KERAS Model
    model = Model (inputs=Activation_0, outputs=X_out)
    
    return model



print('Build the Model')

# Built the Model

model = Attention_Architecture((max_Len_sentence,),embedding_matrix)    
model.summary()

# Learning Parameters 
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    

########################################
## LTSM MODEL : LEARN
########################################

print('Train the Model')

Y_train  = train ['target']

# We are Working with Skewed Classes - Set up Weights in the training

num_ones=Y_train[Y_train==1].shape[0]
num_zeros=Y_train[Y_train==0].shape[0]

d=int(num_zeros/num_ones)


class_weight = {0: 1.,
                1: d}

 
X_train_pad_1 =X_train_pad[10000:]
Y_train_1 = Y_train[10000:]

 
model.fit(X_train_pad_1,Y_train_1,epochs=3,batch_size=256,shuffle=True,class_weight=class_weight)

########################################
## SET F1 ON DEV SET
########################################

X_dev_pad = X_train_pad[:10000]
yp_dev = model.predict(X_dev_pad)
Y_dev = Y_train[:10000]

def set_prediction_threshold (Y,threshold):
    m=Y.shape[0]
    Y_pred=np.zeros((m,),dtype=int)
    for i in range(m):
        if Y[i][0]>threshold:
            Y_pred[i] = 1
        else:
            Y_pred[i] = 0
    return Y_pred

def set_f1_threshold (Y_true,Y_pred_stat,pas):
    F={}
    F1_thres=0
    max_F1=0
    
    for threshold in np.arange (0,1,pas):
        threshold = np.round(threshold, 2)
        Y_pred = set_prediction_threshold(Y_pred_stat,threshold)
        F1=f1_score(Y_true,Y_pred)
        F[threshold]=F1
        
        if F[threshold]>max_F1:
            F1_thres=threshold
            max_F1 = F1
    
    return F1_thres, max_F1, F


F1_thres, Max_F1, F =set_f1_threshold (Y_dev,yp_dev,0.01)


########################################
## PREDICT
########################################

# INPUTS - rappel binary

Y_pred = model.predict(X_test_pad)

Y_pred_bin=set_prediction_threshold(Y_pred,F1_thres)


########################################
## SUBMIT ANSWER IN KAGGLE
########################################
dev_csv = train[:10000]
train_csv = train[:10000]

model.save('model.h5')

# Submission for Kaggle Competition
dev_csv.to_csv('dev.csv',index=False)
train_csv.to_csv('train.csv',index=False)
out_df = pd.DataFrame({'qid':test['qid'].values})
out_df['prediction'] = Y_pred_bin
out_df.to_csv('submission.csv', index=False)
