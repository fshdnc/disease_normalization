#testing turning stuff into vector
import test_run
import numpy
from gensim.models import KeyedVectors
import nltk



"""taken from NLP course material - NER"""

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

pretrained=load_pretrained_word_embeddings(vocabulary, vector_model)

#word embeddings
''' NOT FINISHED
def vectorizer(word,emb):
    """function to vectorize imput word
       emb = 'word_embedding', 'subword_embedding'"""
    if emb = 'word_embedding':
'''

#subword embeddings


## vectorization


##test NCBI corpus, the mentions are not separated by abstract
#list of objects, '/home/lhchan/disease-normalization/data/ncbi-disease/NCBItestset_corpus.txt'
corpus_objects = test_run.load('gitig_truncated_NCBI.txt','NCBI')
#list of mentions
#each mention has a docid and a sections, which contains title and abstract
corpus_mentions = [mention.text for obj in corpus_objects for part in obj.sections for mention in part.mentions]

def tok(text,tokenizer):
    if tokenizer == 'nltk':
        return nltk.word_tokenize(text)
    else:
        raise ValueError('Tokenizer not found!')

def NCBI_tokenizer_and_vectorizer(corpus_mentions,tokenizer):
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

corpus_tokenized_mentions, corpus_vectorized_numpy= NCBI_tokenizer_and_vectorizer(corpus_mentions,'nltk')

##padding
from keras.preprocessing.sequence import pad_sequences
print("Old shape:", corpus_vectorized_numpy.shape)
corpus_vectorized_padded = pad_sequences(corpus_vectorized_numpy, padding='post')
print("New shape:", corpus_vectorized_padded.shape)


#test MEDIC dictionary, '/home/lhchan/disease-normalization/data/ncbi-disease/CTD_diseases.tsv'
dictionary = test_run.load('gitig_truncated_CTD_diseases.tsv','MEDIC')

def MEDIC_dict_tokenizer_and_vectorizer(MEDIC_dict,tokenizer):
    '''construct a new dictionary
       key: canonical ID
       value: list of list of vectorized disease name
       e.g. original_dictionary[id].AllNames: ('1p36.33 deletion', 'Deletion 1p36.33')
            tokenized_dictionary[id]: [['1p36.33', 'deletion'], ['deletion', '1p36.33']]
            vectorized_dictionary[id]: [[1, 1445], [1445, 1]]
    '''
    dictionary_tokenized={}
    dictionary_vectorized = {}
    for i,j in dictionary.items():
        AllNames_tokenized = [tok(i.lower(),'nltk') for i in j.AllNames]
        dictionary_tokenized[i] = AllNames_tokenized
        AllNames_vectorized = [[vocabulary.get(token,1) for token in name] for name in AllNames_tokenized]
        dictionary_vectorized[i] = AllNames_vectorized
    return dictionary_tokenized, dictionary_vectorized

dictionary_tokenized, dictionary_vectorized = MEDIC_dict_tokenizer_and_vectorizer(dictionary,'nltk')

'''
#Visual check of vectorization result of MEDIC
#prints all dict items
for entry in dictionary_vectorized.keys():
    print('mention:',dictionary[entry].AllNames)
    print('tokenized mention',dictionary_tokenized[entry])
    print('vectorized mention:',dictionary_vectorized[entry])
    print('\n')
'''

import candidate_generation
dictionary_processed = candidate_generation.process_MEDIC_dict(dictionary_tokenized,'skipgram')
generated_candidates = candidate_generation.generate_candidate(corpus_tokenized_mentions,dictionary_processed,20)

"""Things to note
have a draft first
go over the theory

what do we put into the system: vector(string, all correct predictions)
how do we train

gensim KeyedVectors objects have not been learned

embedding: try Chiu et al. cambridgeltl
w2v, phrase2vec

vectorization, last resort
lowercase
tokenize: try different tokenizers (note that tokenizer should correspond to that used for word embedding)
if not found in dic, search for all in uppercases as well

masking? refer to pos with features
"""