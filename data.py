# taken from https://github.com/suriyadeepan/practical_seq2seq/blob/master/datasets/cornell_corpus/data.py

UNK = 'unk'
VOCAB_SIZE = 8000

import random
import re
import nltk
import itertools
from collections import defaultdict

import numpy as np
from subprocess import call
import pickle
import word2vec_utils as w2v
from chat_constants import *

''' 
    1. Read from 'movie-lines.txt'
    2. Create a dictionary with ( key = line_id, value = text )
'''
def get_id2line():
    lines = open('data/stripped_movie_lines_results.txt', 'r' ).read().split('\n')
    id2line = {}
    for line in lines:
        _line = line.split(SPLIT_STRING)
        if len(_line) == 4:
            ID = _line[0].strip()
            text = w2v.split_sentence( _line[1] ) #split up special characters i.e. "Hello?!" -> "Hello ? !"
            pos = _line[2]
            neg = _line[3][1:] #[1:] gets rid of the negative sign
            neu = str( 11 - int( pos ) - int(neg) )
            id2line[ID] = " ".join( [ pos, neu, neg, EMOTE_DELIMITER, text ] ) 
    return id2line

'''
    1. Read from 'movie_conversations.txt'
    2. Create a list of [list of line_id's]
'''
def get_conversations():
    conv_lines = open('data/movie_conversations.txt', 'r').read().split('\n')
    convs = [ ]
    for line in conv_lines[:-1]:
        _line = line.split(' ' + SPLIT_STRING + ' ')[-1][1:-1].replace("'","").replace(" ","")
        convs.append(_line.split(','))
    return convs

'''
    1. Get each conversation
    2. Get each line from conversation
    3. Save each conversation to file
'''
def extract_conversations(convs,id2line,path=''):
    idx = 0
    for conv in convs:
        f_conv = open(path + str(idx)+'.txt', 'w')
        for line_id in conv:
            f_conv.write(id2line[line_id])
            f_conv.write('\n')
        f_conv.close()
        idx += 1

'''
    Get lists of all conversations as Questions and Answers
    1. [questions]
    2. [answers]
'''
def gather_dataset(convs, id2line):
    A1 = []; B = []; A2 = [];

    for conv in convs:
        for i in range(len(conv)-2):
            if(conv[i] in id2line and conv[i+1] in id2line and conv[i+2] in id2line ):
                #A1.append(id2line['blank'])
                #B.append(id2line[conv[i]])
                #A2.append(id2line[conv[i+1]])
                #if i + 2 < len(conv):
                A1.append(id2line[conv[i]])
                B.append(id2line[conv[i+1]])
                A2.append(id2line[conv[i+2]])
    
    print( '\nInitially generate ' + str(len(A1)) + ' training data' )
    return np.array(A1),np.array(B),np.array(A2)
           
'''
 filter too long and too short sequences
    return tuple( filtered_ta, filtered_en )
'''
def filter_data(A1_seq, B_seq, A2_seq):
    filtered_A1, filtered_B, filtered_A2 = [], [], []
    data_len = len(A1_seq)

    assert len(A1_seq) == len(B_seq) and len(B_seq) == len(A2_seq)

    j = 0
    for i in range(data_len):
        lengths = map( lambda x: len(x[i].split(' ')), [A1_seq, B_seq, A2_seq ] )
        if( lengths[0] <= MAX_SENT_LENGTH and
            lengths[1] <= MAX_SENT_LENGTH and
            lengths[2] <= MAX_SENT_LENGTH and #A2 can be long because LSTM responds in double lenght sequence 
            #( lengths[0] >= min_sent_length or ( lengths[0] == 1 and A1_seq[i] == '' ) )and
            lengths[0] >= MIN_SENT_LENGTH and
            lengths[1] >= MIN_SENT_LENGTH and
            lengths[2] >= MIN_SENT_LENGTH
          ):
            filtered_A1.append( A1_seq[i] )
            filtered_B.append(  B_seq[i]  )
            filtered_A2.append( A2_seq[i] )
        elif j < 2:
            print( A1_seq[i], B_seq[i], A2_seq[i] )
            print( lengths )
            j+=1

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_A1)
    filtered = int(filt_data_len*100/data_len)
    print( filt_data_len, " is how many pairs remain" )
    print(str(filtered) + '% passed through from original data')

    return np.array(filtered_A1), np.array(filtered_B), np.array(filtered_A2)

'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )
'''
def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ['_'] + [UNK] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist

'''
 filter based on number of unknowns (words not in vocabulary)
  filter out the worst sentences
'''
def filter_unk(A1_w2v,B_w2v, atokenized, w2idx):
    data_len = len(atokenized)

    filtered_A1, filtered_B, filtered_A2 = [], [], []

    for A1_line,B_line,A2_line in zip(A1_w2v,B_w2v, atokenized):
        unk_count_a = len([ w for w in A2_line if w not in w2idx ])
        #TODO allow for non -pristine data 
        if unk_count_a == 0:
            filtered_A1.append(A1_line)
            filtered_B.append(B_line)
            filtered_A2.append(' '.join(A2_line))
                                      
    filtered_A1, filtered_B, filtered_A2 = map( lambda x: np.array( x ), [ filtered_A1, filtered_B, filtered_A2 ] )
    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_A2)
    filtered = int(filt_data_len*100/data_len)
    print(filt_data_len, " lines are now remaining" )
    print(str(filtered) + '% passed through from original data')

    return filtered_A1, filtered_B, filtered_A2

'''
 create the final dataset : 
  - convert list of items to arrays of indices
  - add zero padding
      return ( [array_en([indices]), array_ta([indices]) )
 
'''
def zero_pad(atokenized, w2idx):
    # num of rows
    data_len = len(atokenized)

    # numpy arrays to store indices
    idx_a = np.zeros([data_len, max_sent_length], dtype=np.int32)

    for i in range(data_len):
        a_indices = pad_seq(atokenized[i], w2idx, max_sent_length)
        idx_a[i] = np.array(a_indices)

    return idx_a


'''
 replace words with indices in a sequence
  replace with unknown if word not in lookup
    return [list of indices]
'''
def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))

def seperate_punc(list_of_lines):
    '''for a given line insert spaces between punc and the words'''
    ret = []
    for lines in list_of_lines:
        ret.append( [ re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-])\s*", r" \1", wordlist) for wordlist in lines ] )
    return( ret )

def process_data():
    id2line = get_id2line()
    print('>> gathered id2line dictionary.\n')
    convs = get_conversations()
    print(convs[121:125])
    print('>> gathered conversations.\n')
    A1, B, A2 = gather_dataset(convs,id2line)
    
    print('\n>>saving all lines')
    np.save('All_Lines.npy',np.append(A1,np.append(B,A2)))

    # change to lower case (just for en)
    #A1, B, A2 = [ [ line.lower() for line in sentences ] for sentences in [A1, B, A2 ] ]
    #A1, B, A2 = seperate_punc( [ A1, B, A2 ] )
    
    # filter out too long or too short sequences
    print('\n>> 2nd layer of filtering')
    A1, B, A2 = filter_data(A1,B,A2)
     
    #print('\n>> Segment lines into words')
    #A2_tokenized = [ [w.strip() for w in wordlist.split(' ') if w] for wordlist in A2_lines ]
                 
    #print('\n:: Sample from segmented list of words')
    #for a in A2_tokenized[141:145]:
    #    print(a)

    # indexing -> idx2w, w2idx 
    #print('\n >> Index words')
    #idx2w, w2idx, freq_dist = index_( A2_tokenized, vocab_size=VOCAB_SIZE)
    
    # filter out sentences with too many unknowns
    #print('\n >> Filter Unknowns')
    #filtered_A1, filtered_B, filtered_A2 = filter_unk(A1_lines, B_lines, A2_tokenized, w2idx)
    #print('\n Final dataset len : ' + str(len(filtered_A2)))

    #print('\n >> Zero Padding')
    #padded_A2 = zero_pad(filtered_A2, w2idx)

    
                
    print('\n >> Save numpy arrays to disk')
    # save them
    np.save('A1.npy', A1)
    np.save('B.npy',  B )
    np.save('A2.npy', A2)

    # let us now save the necessary dictionaries
    #metadata = {
    #        'w2idx' : w2idx,
    #        'idx2w' : idx2w,
    #        'limit' : [min_sent_length,max_sent_length],
    #        'freq_dist' : freq_dist
    #         }

    # write to disk : data control dictionaries
    #with open('metadata.pkl', 'wb') as f:
    #    pickle.dump(metadata, f)

    # count of unknowns
    # unk_count = (padded_A2 == 1).sum()
    # count of words
    # word_count = (padded_A2 > 1).sum()

    # print('% unknown : {0}'.format(100 * (unk_count/word_count)))
    #print('Dataset count : ' + str(filtered_A2.shape[0]))


    print('>> gathered questions and answers.\n')
    #prepare_seq2seq_files(questions,answers)


if __name__ == '__main__':
    process_data()

def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    A1 = np.load(PATH + 'A1.npy' )
    B  = np.load(PATH + 'B.npy'  )
    A2 = np.load(PATH + 'A2.npy' )
  
    return A1, B, A2


def categorical_representation(vector_rep):
    ret = np.zeros((VOCAB_SIZE + 2, len(vector_rep)))
    for i,word_index in enumerate(vector_rep):
        ret[word_index][i] = 1
    return(ret)
    

