#!/usr/bin/env python3

from collections import Counter
import math
import configparser as cp
config = cp.ConfigParser(strict=False)
config.read('defaults.cfg')

'''
Things to be fixed
1. skip-gram generation
2. cutoff score for candidate generation

'''

'''
1. function to turn tokenized MEDIC dict into skip-grammed MEDIC dict
2. function to calculate similarity score for a term
3. function to generate skipgram from string or list
4. function to calculate cosine similarity
5. function to generate candidates
'''

def process_MEDIC_dict(tokenized_MEDIC_dict,method):
    '''
       method: 'skipgram'
       construct a new dictionary
       key: canonical ID
       value: list of list of processed mention
       e.g. original_dictionary[id].AllNames: [['1p36.33', 'deletion'], ['deletion', '1p36.33']]
            processed_dictionary[id]: [['1p', '36', '.3', 'de', 'le', 'ti', 'on'], ['de', 'le', 'ti', 'on', '1p', '36', '.3']]
    '''
    dictionary_processed = {}
    if method == 'skipgram':
        for i,j in tokenized_MEDIC_dict.items():
            #requires config
            AllNames_skipgram = [generate_skipgram(name,config.getint('ngram','n'),config.getint('ngram','s')) for name in j]
            print(j,AllNames_skipgram)
            print('\n')
            dictionary_processed[i] = AllNames_skipgram
    return dictionary_processed

#dictionary_processed = process_MEDIC_dict(dictionary_tokenized,'skipgram')

## cosine similarity of skip-grams
## idea taken from tzlink

def term_sim(mention,candidates):
    '''
    mention: skip-grammed mention as a list
    candidates: list of candidates, each a skip-grammed term
    output: highest score of the list of candidates
    '''
    sim_score = []
    for candidate in candidates:
        sim = cosine_similarity_ngrams(mention,candidate)
        sim_score.append(sim)
    return max(sim_score)

def generate_skipgram(w_or_v,n,s):
    '''
    input: single token/list of tokens
    output: list of skip-grams
    n: n-gram
    s: only print the k*s-th ngram
    designed for single tokens, does not take into account
    spaces, does not add space in front/at the end of the
    token
    '''
    def check_pos_int(x,message):
        if x <= 0 or not isinstance(x,int):
            raise ValueError(message)
    check_pos_int(n,'Value for n must be a positive integer')
    check_pos_int(s+1,'Value for s must be non-negative integer.')
    if isinstance(w_or_v,str):
        if ' ' in w_or_v:
            raise ValueError('String contains space characters.')
        skipgrams=[w_or_v[i:i+n] for i in range(0,len(w_or_v)-(n-1),s+1)]
    elif isinstance(w_or_v,list):
        skipgrams = []
        for token in w_or_v:
            token_skipgrams=[token[i:i+n] for i in range(0,len(token)-(n-1),s+1)]
            skipgrams = skipgrams + token_skipgrams
    else:
        raise TypeError('Variable w_or_v must be string or list.')
    return skipgrams

def cosine_similarity_ngrams(a, b):
    '''
    taken from: https://gist.github.com/gaulinmp/da5825de975ed0ea6a24186434c24fe4
    '''
    vec1 = Counter(a)
    vec2 = Counter(b)    
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    return float(numerator) / denominator

def generate_candidate(tokenized_mentions,dictionary,n):
    '''inputs:
          tokenized_mentions: list of lists (tokenized mentions)
                              e.g. corpus_tokenized_mentions
          dictionary: skip-grammed dictionary terms
                      e.g. dictionary_processed
       output: list of lists of n (key,score) tuples
       1. go through every tokenized mention
       2. turn each mention into skipgram
       3. compare with skip-grammed dictionary mention
       4. return n highest scoring dictionary terms
    '''
    generated_candidates = []
    for mention in tokenized_mentions:
        mention_skipgram = generate_skipgram(mention,config.getint('ngram','n'),config.getint('ngram','s'))
        candidate_score = []
        for key,allnames in dictionary.items():
            score = term_sim(mention_skipgram,allnames)
            candidate_score.append((key,score))
        candidate_score = sorted(candidate_score, key=lambda x: x[1],reverse=True)
        generated_candidates.append(candidate_score[:n])
    return generated_candidates

#generate_candidate(corpus_tokenized_mentions,dictionary_processed,20)

#print(token,inversed_vocabulary[token],KeyedVectors.word_vec(vector_model,inversed_vocabulary[token]))