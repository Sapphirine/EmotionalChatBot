import numpy as np

BLANK = np.array( [0]*100 + [1] )
SPLIT_STRING = '+++$+++'
EMOTE_DELIMITER = '@'
BLANK_LINE = " ".join( [ '1', '9', '1', EMOTE_DELIMITER, '' ] )
MIN_SENT_LENGTH = 10
MAX_SENT_LENGTH = 30
EMBED_DIM = len(BLANK)
VOCAB_SIZE = 10000
UNK = VOCAB_SIZE + 1
NULL = VOCAB_SIZE + 2