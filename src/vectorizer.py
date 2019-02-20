#!/usr/bin/env python3

#import load
import numpy
from gensim.models import KeyedVectors
import nltk
import tzlink


"""taken from NLP course material - NER"""

def prepare_embedding_vocab(filename, binary = True, limit = 50000):
    '''filename: '~/disease-normalization/data/embeddings/wvec_50_haodi-li-et-al.bin'
       1. Use gensim for reading in embedding model
       2. Sort based on the index to make sure that they are in the correct order
       3. Normalize the vectors
       4. Build vocabulary mappings, zero for padding
       5. Create an inverse dictionary
    '''
    vector_model = KeyedVectors.load_word2vec_format(filename, binary = binary, limit = limit)
    #vector_model=KeyedVectors.load_word2vec_format(config['embedding']['emb_file'], binary=True, limit=50000)
    words = [k for k,v in sorted(vector_model.vocab.items(),key = lambda x:x[1].index)]
    vector_model.init_sims(replace = True)
    vocabulary={"<SPECIAL>": 0, "<OOV>": 1}
    for word in words:
        vocabulary.setdefault(word, len(vocabulary))
    inversed_vocabulary={value:key for key, value in vocabulary.items()}
    return vector_model, vocabulary, inversed_vocabulary

#vector_model, vocabulary, inversed_vocabulary = prepare_embedding_vocab('~/disease-normalization/data/embeddings/wvec_50_haodi-li-et-al.bin', binary = True, limit = 50000)

'''
#gensim for reading in embedding model
vector_model = KeyedVectors.load_word2vec_format('~/disease-normalization/data/embeddings/wvec_50_haodi-li-et-al.bin', binary = True, limit = 50000)

##sort based on the index to make sure that they are in the correct order
words = [k for k,v in sorted(vector_model.vocab.items(),key = lambda x:x[1].index)]
#print('Words from embedding model:',len(words))
#print('First 50 words:',words[:50])

##Normalize the vectors
#print('Before normalization:',vector_model.get_vector('in')[:10])
vector_model.init_sims(replace = True)
#print('After normalization:',vector_model.get_vector('in')[:10])

# Build vocabulary mappings
vocabulary={"<SPECIAL>": 0, "<OOV>": 1} # zero has a special meaning in sequence models, prevent using it for a normal word
for word in words:
    vocabulary.setdefault(word, len(vocabulary))

#print("Words in vocabulary:",len(vocabulary))
inversed_vocabulary={value:key for key, value in vocabulary.items()} # inverse the dictionary
'''

def load_pretrained_word_embeddings(vocab,embedding_model):
    """vocab: vocabulary from data vectorizer
       embedding_model: model loaded with gensim"""
    pretrained_embeddings = numpy.random.uniform(low=-0.05, high=0.05, size=(len(vocab)-1,embedding_model.vectors.shape[1]))
    pretrained_embeddings = numpy.vstack((numpy.zeros(shape=(1,embedding_model.vectors.shape[1])), pretrained_embeddings))
    found=0
    for word,idx in vocab.items():
        if word in embedding_model.vocab:
            pretrained_embeddings[idx]=embedding_model.get_vector(word)
            found+=1           
    print("Found pretrained vectors for {found} words.".format(found=found))
    return pretrained_embeddings

#pretrained = load_pretrained_word_embeddings(vocabulary, vector_model)

#word embeddings
''' NOT FINISHED
def vectorizer(word,emb):
    """function to vectorize imput word
       emb = 'word_embedding', 'subword_embedding'"""
    if emb = 'word_embedding':
'''

#subword embeddings


## vectorization

'''
##test NCBI corpus, the mentions are not separated by abstract
#list of objects, '/home/lhchan/disease-normalization/data/ncbi-disease/NCBItestset_corpus.txt'
corpus_objects = load.load('gitig_truncated_NCBI.txt','NCBI')
#list of mentions
#each mention has a docid and a sections, which contains title and abstract
corpus_mentions = [mention.text for obj in corpus_objects for part in obj.sections for mention in part.mentions]
'''

def tok(text,tokenizer):
    if tokenizer == 'nltk':
        return nltk.word_tokenize(text)
    elif tokenizer == 'tzlink':
        return tzlink.tzlink_tokenizer(text)
    else:
        raise ValueError('Tokenizer not found!')

def NCBI_tokenizer_and_vectorizer(vocabulary,corpus_mentions,tokenizer):
    """
    input: list of mentions, name of tokenizer
    returns:1. list of tokenized mentions
            2. 2D numpy array of lowercased, tokenized, vectorized mentions
    1. lowercase the mentions
    2. tokenize the mentions (using nltk for now because of NER example)
    3. vectorize the mentions
    4. turn the list into numpy array
    """
    corpus_tokenized_mentions = [tok(text.lower(),tokenizer) for text in corpus_mentions]
    corpus_vectorized_mentions = [[vocabulary.get(text,1) for text in mention] for mention in corpus_tokenized_mentions]
    corpus_vectorized_numpy = numpy.array(corpus_vectorized_mentions)
    return corpus_tokenized_mentions, corpus_vectorized_numpy

#corpus_tokenized_mentions, corpus_vectorized_numpy= NCBI_tokenizer_and_vectorizer(corpus_mentions,'nltk')

'''
##padding
from keras.preprocessing.sequence import pad_sequences
print("Old shape:", corpus_vectorized_numpy.shape)
corpus_vectorized_padded = pad_sequences(corpus_vectorized_numpy, padding='post')
print("New shape:", corpus_vectorized_padded.shape)
'''

'''
#test MEDIC dictionary, '/home/lhchan/disease-normalization/data/ncbi-disease/CTD_diseases.tsv'
#dictionary = load.load('/home/lhchan/disease-normalization/data/ncbi-disease/CTD_diseases.tsv','MEDIC')
dictionary = load.load('gitig_truncated_CTD_diseases.tsv','MEDIC')
'''

#def MEDIC_dict_tokenizer(MEDIC_dict,tokenizer,vocabulary):
def MEDIC_dict_tokenizer_and_vectorizer(MEDIC_dict,tokenizer,vocabulary):
    '''construct a new dictionary
       key: canonical ID
       value: list of list of vectorized disease name
       e.g. original_dictionary[id].AllNames: ('1p36.33 deletion', 'Deletion 1p36.33')
            tokenized_dictionary[id]: [['1p36.33', 'deletion'], ['deletion', '1p36.33']]
            #vectorized_dictionary_np[id]: numpy.array([[1, 1445], [1445, 1]])
    '''
    dictionary_tokenized={}
    #dictionary_vectorized_np = {}
    dictionary_vectorized = {}
    for i,j in MEDIC_dict.items():
        AllNames_tokenized = [tok(i.lower(),'nltk') for i in j.AllNames]
        dictionary_tokenized[i] = AllNames_tokenized
        '''
        #vectorized in np.array format
        AllNames_vectorized_np = numpy.array([numpy.array([vocabulary.get(token,1) for token in name]) for name in AllNames_tokenized])
        dictionary_vectorized_np[i] = AllNames_vectorized_np
    return dictionary_tokenized, dictionary_vectorized_np
        '''
        
        AllNames_vectorized = [[vocabulary.get(token,1) for token in name] for name in AllNames_tokenized]
        dictionary_vectorized[i] = AllNames_vectorized
        
    return dictionary_tokenized, dictionary_vectorized

#dictionary_tokenized, dictionary_vectorized = MEDIC_dict_tokenizer_and_vectorizer(dictionary,'nltk')

'''
#decided to just vectorize the whole dict
def candidate_vectorizer(tokenized_candidate,tokenizer):
    pass
'''

'''
#Visual check of vectorization result of MEDIC
#prints all dict items
for entry in dictionary_vectorized.keys():
    print('mention:',dictionary[entry].AllNames)
    print('tokenized mention',dictionary_tokenized[entry])
    print('vectorized mention:',dictionary_vectorized[entry])
    print('\n')
'''

'''
import candidate_generation
dictionary_processed = candidate_generation.process_MEDIC_dict(dictionary_tokenized,'skipgram')
generated_candidates = candidate_generation.generate_candidate(corpus_tokenized_mentions,dictionary_processed,20)
'''

def MEDIC_dict_tokenizer_no_cangen(MEDIC_dict,tokenizer):
    '''construct a new dictionary
       key: canonical ID
       value: tokenized disease name
       e.g. original_dictionary[id].AllNames: ('1p36.33 deletion', 'Deletion 1p36.33')
            tokenized_dictionary[id]: [['1p36.33', 'deletion'], ['deletion', '1p36.33']]
       input MEDIC_dict value: (DiseaseID,DiseaseName,AllDiseaseIDs,AllNames)
    '''
    dictionary_tokenized={}
    for i,j in MEDIC_dict.items():
        dictionary_tokenized[i] = tok(j.DiseaseName.lower(),'nltk')
    return dictionary_tokenized

def MEDIC_dict_vectorizer_no_cangen(tokenized_dict,vocabulary):
    '''construct a new dictionary
       key: canonical ID
       value: list vectorized disease name
       e.g. original_dictionary[id].AllNames: ('1p36.33 deletion', 'Deletion 1p36.33')
            tokenized_dictionary[id]: [['1p36.33', 'deletion'], ['deletion', '1p36.33']]
            #vectorized_dictionary_np[id]: numpy.array([[1, 1445], [1445, 1]])
    '''
    dictionary_vectorized = {}
    for i,j in tokenized_dict.items():
        dictionary_vectorized[i] = [vocabulary.get(token,1) for token in j]        
    return dictionary_vectorized

def MEDIC_dict_untokenized(MEDIC_dict,allnames):
    '''construct a new dictionary
       key: canonical ID
       value: untokenized disease name, either (1) only canonical name (string) or (2) list of all names
    '''
    dictionary={}
    if not allnames: # use only the canonical name
        for i,j in MEDIC_dict.items():
            dictionary[i] = j.DiseaseName
    else:
        for i,j in MEDIC_dict.items():
            dictionary[i] = list(j.AllNames)
            assert len(dictionary[i]) == len(j.AllNames)
    return dictionary