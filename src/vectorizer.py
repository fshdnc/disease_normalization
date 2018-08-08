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
#list of objects
corpus_objects = test_run.load('gitig_truncated_NCBI.txt','NCBI')
#list of mentions
#each mention has a docid and a sections, which contains title and abstract
corpus_mentions = [mention.text for obj in corpus_objects for part in obj.sections for mention in part.mentions]

def NCBI_vectorizer(corpus_mentions):
    """
    input: list of mentions
    returns: 2D numpy array of lowercased, tokenized, vectorized mentions
    1. lowercase the mentions
    2. tokenize the mentions (using nltk for now because of NER example)
    3. vectorize the mentions
    4. turn the list into numpy array
    """
    corpus_lowercased_mentions = [text.lower() for text in corpus_mentions]
    corpus_tokenized_mentions = [nltk.word_tokenize(text) for text in corpus_lowercased_mentions]
    corpus_vectorized_mentions = [[vocabulary.get(text,1) for text in mention] for mention in corpus_tokenized_mentions]
    return numpy.array(corpus_vectorized_mentions)

corpus_vectorized_numpy = NCBI_vectorizer(corpus_mentions)

##padding
from keras.preprocessing.sequence import pad_sequences
print("Old shape:", corpus_vectorized_numpy.shape)
corpus_vectorized_padded = pad_sequences(corpus_vectorized_numpy, padding='post')
print("New shape:", corpus_vectorized_padded.shape)


#test MEDIC dictionary
dictionary = test_run.load('gitig_truncated_CTD_diseases.tsv','MEDIC')

def vectorize_MEDIC_dict(MEDIC_dict,tokenizer):
    '''construct a new dictionary
       key: canonical ID
       value: list of list of vectorized disease name
       e.g. original_dictionary[id].AllNames: ('1p36.33 deletion', 'Deletion 1p36.33')
            vectorized_dictionary[id]: [[1, 1445], [1445, 1]]
    '''
    dictionary_vectorized = {}
    if tokenizer == 'nltk':
        for i,j in dictionary.items():
            AllNames_tokenized = [nltk.word_tokenize(i.lower()) for i in j.AllNames]
            AllNames_vectorized = [[vocabulary.get(token,1) for token in name] for name in AllNames_tokenized]
            dictionary_vectorized[i] = AllNames_vectorized
    else:
        print('Tokenizer not recognized.')
    return dictionary_vectorized

dictionary_vectorized = vectorize_MEDIC_dict(dictionary,'nltk')

'''
#Visual check of vectorization result of MEDIC
#prints all dict items
for entry in dictionary_vectorized.keys():
    print('mention:',dictionary[entry].AllNames)
    print('vectorized mention:',dictionary_vectorized[entry])
    print('\n')
'''


#have a draft first
#go over the theory

#what do we put into the system: vector(string, all correct predictions)
#how do we train

"""Things to note
gensim KeyedVectors objects have not been learned

embedding: try Chiu et al. cambridgeltl
w2v, phrase2vec

vectorization, last resort
lowercase
tokenize: try different tokenizers (note that tokenizer should correspond to that used for word embedding)
if not found in dic, search for all in uppercases as well

masking? refer to pos with features
"""

#candidate generation
for mention in corpus_vectorized_numpy:
    for token in mention:
        #print(token,inversed_vocabulary[token],KeyedVectors.word_vec(vector_model,inversed_vocabulary[token]))
        