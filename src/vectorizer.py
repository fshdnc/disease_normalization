#testing turning stuff into vector
import test_run
import numpy
from gensim.models import KeyedVectors
import nltk



"""taken from NLP course material - NER"""

#gensim for reading in embedding model
vector_model = KeyedVectors.load_word2vec_format('~/disease-normalization/data/embeddings/wvec_50_haodi-li-et-al.bin', binary = True, limit = 50000)

#sort based on the index to make sure that they are in the correct order
words = [k for k,v in sorted(vector_model.vocab.items(),key = lambda x:x[1].index)]
print('Words from embedding model:',len(words))
print('First 50 words:',words[:50])

#Normalize the vectors
print('Before normalization:',vector_model.get_vector('in')[:10])
vector_model.init_sims(replace = True)
print('After normalization:',vector_model.get_vector('in')[:10])

# Build vocabulary mappings
vocabulary={"<SPECIAL>": 0, "<OOV>": 1} # zero has a special meaning in sequence models, prevent using it for a normal word
for word in words:
    vocabulary.setdefault(word, len(vocabulary))
print("Words in vocabulary:",len(vocabulary))
inversed_vocabulary={value:key for key, value in vocabulary.items()} # inverse the dictionary

def load_pretrained_word_embeddings(vocab,embedding_model):
    """vocab: vocabulary from data vectorizer
       embedding_model: model loaded with gensim"""
    pretrained_embeddings=numpy.random.uniform(low=-0.05, high=0.05, size=(len(vocab)-1,embedding_model.vectors.shape[1]))
    pretrained_embeddings = numpy.vstack((numpy.zeros(shape=(1,embedding_model.vectors.shape[1])), pretrained_embeddings))
    found=0
    for word,idx in vocab.items():
        if word in embedding_model.vocab:
            pretrained_embeddings[idx]=embedding_model.get_vector(word)
            found+=1           
    print("Found pretrained vectors for {found} words.".format(found=found))
    return pretrained_embeddings

pretrained=load_pretrained_word_embeddings(vocabulary, vector_model)

"""
#vectorizing data
import numpy
def vectorizer(vocab, texts, label_map, labels):
    vectorized_data = [] # turn text into numbers based on our vocabulary mapping
    vectorized_labels = [] # same thing for the labels
    sentence_lengths = [] # Number of tokens in each sentence
    
    for i, one_example in enumerate(texts):
        vectorized_example = []
        vectorized_example_labels = []
        for word in one_example:
            vectorized_example.append(vocab.get(word, 1)) # 1 is our index for out-of-vocabulary tokens
        
        for label in labels[i]:
            vectorized_example_labels.append(label_map[label])

        vectorized_data.append(vectorized_example)
        vectorized_labels.append(vectorized_example_labels)
        
        sentence_lengths.append(len(one_example))
        
    vectorized_data = numpy.array(vectorized_data) # turn python list into numpy matrix
    vectorized_labels = numpy.array(vectorized_labels)
    
    return vectorized_data, vectorized_labels, sentence_lengths

vectorized_data, vectorized_labels, lengths=vectorizer(vocabulary, texts, label_map, labels)
validation_vectorized_data, validation_vectorized_labels, validation_lengths=vectorizer(vocabulary, validation_texts, label_map, validation_labels)
"""

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
    2. tokenize the mentions
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
'''construct a new dictionary
   key: canonical ID
   value: list of list of vectorized disease name
   e.g. original_dictionary[id].AllNames: ('1p36.33 deletion', 'Deletion 1p36.33')
        vectorized_dictionary[id]: [[1, 1445], [1445, 1]]
'''
dictionary_vectorized = {}
for i,j in dictionary.items():
    AllNames_tokenized = [nltk.word_tokenize(i.lower()) for i in j.AllNames]
    AllNames_vectorized = [[vocabulary.get(token,1) for token in name] for name in AllNames_tokenized]
    dictionary_vectorized[i] = AllNames_vectorized


#have a draft first
#go over the theory

#what do we put into the system: vector(string, all correct predictions)
#how do we train

"""Things to note
gensim KeyedVectors objects have not been learned

biomedical domain tokenizer paper to be read

vectorization, last resort
lowercase
tokenize
if not found in dic, search for all in uppercases as well

masking? refer to pos with features
"""