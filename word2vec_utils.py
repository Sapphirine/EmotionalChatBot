import gensim
import re
import numpy as np

word2vec_path = './GoogleNews-vectors-negative300.bin'
mapped_words = { '!':'exclamation',
                    '?':'question',   
                    ',':'comma',
                    '.':'period',
                    ':':'colon',
                    ';':'semicolon',
                    "'":'`',
                    '`em':'em',
                   '-':'hyphen',
                  }

unknown_words = {}

def initialize():
    return gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

# input: trained model, one sentence (string)
# output: a list of word2vec vectors for that sentence
def vectorize(model, sentence, pad_length = -1):
    if model is None:
        print( "Please initialize the model before vectorizing a sentence!" )
        return( [] )

    sentence = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-])\s*", r" \1", sentence)
    words = str.split(sentence.strip(), " ")
    vectorized_sentence = []

    for word in words:
        if word == "":
            continue
        if word in model:
            vectorized_sentence.append(model[word])
        elif word in mapped_words:
            vectorized_sentence.append(model[mapped_words[word]])
        else:
            # Yoon Kim's unknown word handling, 2014
            # https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py#L88
            if word not in unknown_words:
                unknown_words[word] = np.random.uniform(-0.25,0.25,300)
            vectorized_sentence.append(unknown_words[word])   

    if( pad_length > 0 ):
        while( len( vectorized_sentence ) < pad_length ):
            vectorized_sentence.append(np.array([1] + [0]*299)) #Vector pointing entirely along first dimension
            
    return( np.array(vectorized_sentence) )

#model = initialize()
#print vectorize(model, "Hello, this is a test!")
