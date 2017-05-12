import gensim
import re
import numpy as np
import json
import pickle
import math
import data
from chat_constants import *
from gensim.models import Word2Vec

word2vec_path = './GoogleNews-vectors-negative300.bin'

word2vec_path = './GoogleNews-vectors-negative300.bin'
unknown_path  = 'unknown_words_stored.pkl'

def initialize():
    global w2v_model
    try:
        w2v_model
    except:
        w2v_model = Word2Vec.load('movie_trained_w2v_model')#gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)#
    return( w2v_model ) 

def get_unknown_vectors():
    with open(unknown_path, 'rb') as f:
        unknown_vectors = pickle.load(f)
    return( unknown_vectors )

def save_unknown_vectors(vecs):
    with open(unknown_path, 'wb') as f:
        pickle.dump(vecs, f)

def unvectorize_initialize():
    reset_global_unknown_vectors()
    initialize_global_w2v_model()
    
def unvectorize_sentence( sentence ):
    ''' Takes a numpy array indicating a sentence where each entry is a 301 dim
        vector indicating the predicted embedding of a word.
        
        returns a single string representing the sentence
    '''
    words = []
    for word in sentence:
        words.append( unvectorize_word( word ) )
    return( " ".join(words) )

def unvectorize_word( word ):
    ''' Takes a numpy array inidicating a word embedid in 301 dim space and 
        returns the most likely word
    '''
    if word[-1] == 1: #Parse NULL predicted words
        ret = "_"
    else:
        # Get w2v best word and similarity score
        word = word[:-1]
        word_sum_square = np.sum(word**2) # calculate once here to avoid multiple calculation
        w2v_word,w2v_similarity = w2v_model.similar_by_vector( word, topn = 1 )[0]
        #Very confident it is this word
        if w2v_similarity > 0.999:
            ret = w2v_word
        else:
            #Unknown word 
            uk_word,uk_similarity = get_best_unknown_word( word, word_sum_square )
            print( "w2v word-score", w2v_word, w2v_similarity, "\n", "uk word-score", uk_word, uk_similarity )
            if w2v_similarity > uk_similarity:
                ret = w2v_word
            else:
                ret = uk_word
            
    return ret
        
def get_best_unknown_word( word, word_sum_square ):
    global unknown_vectors
    best_similarity_so_far = 0
    best_word_so_far = "_"
    for uk_word in unknown_vectors:
        uk_embedding = unknown_vectors[uk_word][:-1]#remove null dimension
        similarity = get_similarity( uk_embedding, word, word_sum_square )
        if similarity > best_similarity_so_far:
            best_similarity_so_far = similarity
            best_word_so_far = uk_word
            
    return( best_word_so_far, best_similarity_so_far )
        
def get_similarity( w1,w2,sumyy ):
    sumxx = np.sum(w1**2 )
    sumxy = np.sum(w1 * w2 )
    ret = sumxy/math.sqrt(sumxx*sumyy)
    return( ret )
# input: trained model, one sentence (string)
# output: a list of word2vec vectors for that sentence

def split_sentence( sentence ):
    ''' takes a string and returns a string with the punctuation seperated out 
        split up special characters i.e. "Hello?!" -> "Hello ? !"
    '''
    #sentence = sentence.encode('ascii','ignore')
    sentence = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-])\s*", r" \1", sentence)
    sentence = " ".join(sentence.split()) #Remove multiple spaces
    return sentence 

def vectorize( sentence, pad_length = -1, model = None ):
    global unknown_vectors
    global w2v_model

    if model is None:
        print("Attempting to use global model")
        model = w2v_model
        
    model_dimension = len(model['the'])#Assume the to always be in the model

    sentence = split_sentence( sentence )
    words = sentence.split(" ")
    vectorized_sentence = []

    #should_save_unknown_vectors = False
    for word in words:
        lower_word = word.lower()
        if word in model:
            vectorized_sentence.append(np.append(model[word],0))
        elif lower_word in model:
            vectorized_sentence.append(np.append(model[lower_word],0))
        elif lower_word in unknown_vectors:
            vectorized_sentence.append(unknown_vectors[lower_word])
        else:
            # Yoon Kim's unknown word handling, 2014
            # https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py#L88
            unknown_vectors[lower_word] = np.append(np.random.uniform(-0.25,0.25,EMBED_DIM-1),0)
            vectorized_sentence.append(unknown_vectors[lower_word])
            #should_save_unknown_vectors = True
            
    if( pad_length != -1 ):
        while( len(vectorized_sentence) < pad_length ):
            vectorized_sentence.append(BLANK)
      
    return np.array(vectorized_sentence)

def one_hot_vectorize( sentence, pad_length = -1, word_freqs = None ):
    if word_freqs is None:
        print( 'Loaded word_frequencies data from disk' )
        word_freqs = np.load('words_in_order_of_freq.npy')
    
    word_freqs = word_freqs.tolist()
    sentence = split_sentence( sentence ).lower()
    words = sentence.split(" ")
    vectorized_sentence = []
    
    for word in words:
        try:
            number = word_freqs.index(word)
            if number > VOCAB_SIZE:
                number = UNK
        except:
            number = UNK
        vectorized_sentence.append( number )

    if( pad_length != -1 ):
        while( len(vectorized_sentence) < pad_length ):
            vectorized_sentence.append(NULL)
    
    return np.array(vectorized_sentence).astype('int16')

def one_hot_unvectorize( sentence, word_freqs = None ):
    if word_freqs is None:
        print( 'Loaded word_frequencies data from disk' )
        word_freqs = np.load('words_in_order_of_freq.npy') 
        #print(word_freqs)
    pred_words = []
    #for word_vec in sentence[0]:
    #    pred_words.append( word_freqs[np.where(word_vec==max(word_vec))[0][0]] )
    for i in range(30):
        pred_words.append(word_freqs[sentence[0,i]].decode('utf-8'))
    print( " ".join(pred_words))

def get_training_data(A1=None,B=None,A2=None,model=None):
    ''' gets embeded versions of data stored on disk or passed in as A1,B,A2
        to get the ones stored on disk be sure to run data.py first 
    '''
    global w2v_model
    if A1 is None:
        A1,B,A2 = data.load_data()
    if model is None:
        model = w2v_model
    A1_train = []
    B_train = []
    A2_train = []
    i = 0
    reset_global_unknown_vectors()
    for i in range(len(A1)):
        A1_train.append( vectorize(A1[i],pad_length = MAX_SENT_LENGTH,model = model) )
        B_train.append( vectorize(B[i],pad_length = MAX_SENT_LENGTH,model = model ) )
        A2_train.append( vectorize(A2[i],pad_length = MAX_SENT_LENGTH,model = model) )
    save_global_unknown_vectors()
    return np.array(A1_train),np.array(B_train),np.array(A2_train)

def get_training_data_one_hot_out(A1=None,B=None,A2=None,model=None):
    ''' gets embeded versions of data stored on disk or passed in as A1,B,A2
        to get the ones stored on disk be sure to run data.py first 
    '''
    global w2v_model
    if A1 is None:
        A1,B,A2 = data.load_data()
    if model is None:
        model = w2v_model
    A1_train = []
    B_train = []
    A2_train = []
    i = 0
    reset_global_unknown_vectors()
    
    word_freqs = np.load('words_in_order_of_freq.npy')

    for i in range(len(A1)):
        A1_train.append( vectorize(A1[i],pad_length = MAX_SENT_LENGTH,model = model) )
        B_train.append( vectorize(B[i],pad_length = MAX_SENT_LENGTH,model = model ) )
        A2_train.append( one_hot_vectorize(A2[i],pad_length = MAX_SENT_LENGTH,word_freqs = word_freqs) )
    save_global_unknown_vectors()
    return np.array(A1_train),np.array(B_train),np.array(A2_train)

def reset_global_unknown_vectors():
    global unknown_vectors 
    unknown_vectors = get_unknown_vectors()

def save_global_unknown_vectors():
    global unknown_vectors 
    save_unknown_vectors(unknown_vectors)

def initialize_global_w2v_model():
    try:
        w2v_model
    except:
        initialize()
    
