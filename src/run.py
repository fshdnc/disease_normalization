import vectorizer
import load
import configparser as cp

config = cp.ConfigParser()
config.read('defaults.cfg')


vector_model, vocabulary, inversed_vocabulary = vectorizer.prepare_embedding_vocab('~/disease-normalization/data/embeddings/wvec_50_haodi-li-et-al.bin', binary = True, limit = 50000)
pretrained = vectorizer.load_pretrained_word_embeddings(vocabulary, vector_model)

##test NCBI corpus, the mentions are not separated by abstract
#list of objects, '/home/lhchan/disease-normalization/data/ncbi-disease/NCBItestset_corpus.txt'
corpus_objects = load.load(config['corpus']['corpus_file'],'NCBI')
#list of mentions
#each mention has a docid and a sections, which contains title and abstract
corpus_mentions = [mention.text for obj in corpus_objects for part in obj.sections for mention in part.mentions]

corpus_tokenized_mentions, corpus_vectorized_numpy = vectorizer.NCBI_tokenizer_and_vectorizer(vocabulary,corpus_mentions,'nltk')

##padding
from keras.preprocessing.sequence import pad_sequences
print("Old shape:", corpus_vectorized_numpy.shape)
corpus_vectorized_padded = pad_sequences(corpus_vectorized_numpy, padding='post')
print("New shape:", corpus_vectorized_padded.shape)

##test MEDIC dictionary
dictionary = load.load(config['terminology']['dict_file'],'MEDIC')

dictionary_tokenized, dictionary_vectorized = vectorizer.MEDIC_dict_tokenizer_and_vectorizer(dictionary,'nltk',vocabulary)

import candidate_generation
dictionary_processed = candidate_generation.process_MEDIC_dict(dictionary_tokenized,'skipgram')
generated_candidates = candidate_generation.generate_candidate(corpus_tokenized_mentions,dictionary_processed,config['candidate']['n'])

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