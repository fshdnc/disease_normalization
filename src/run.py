import vectorizer
import load
import configparser as cp
import args
import logging

#configurations
config = cp.ConfigParser(strict=False)
config.read('defaults.cfg')

#argparser
args = args.get_args()
'''
>>> args.train
False
'''

#logging
logger = logging.getLogger(__name__)
logger.setLevel(config.getint('settings','logging_level'))
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
file_handler = logging.FileHandler(config['settings']['logging_filename'])
file_handler.setLevel(config.getint('settings','logging_level'))
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

#word embedding
logger.info('Preparing word embeddings...')
vector_model, vocabulary, inversed_vocabulary = vectorizer.prepare_embedding_vocab('~/disease-normalization/data/embeddings/wvec_50_haodi-li-et-al.bin', binary = True, limit = 50000)
pretrained = vectorizer.load_pretrained_word_embeddings(vocabulary, vector_model)

#NCBI corpus
##test NCBI corpus, the mentions are not separated by abstract
#list of objects, '/home/lhchan/disease-normalization/data/ncbi-disease/NCBItestset_corpus.txt'
logger.info('Loading NCBI corpus...')
corpus_objects = load.load(config['corpus']['corpus_file'],'NCBI')
#list of mentions
#each mention has a docid and a sections, which contains title and abstract
corpus_mentions = [mention.text for obj in corpus_objects for part in obj.sections for mention in part.mentions]
logger.info('Tokenizing and vectorizing mentions...')
corpus_tokenized_mentions, corpus_vectorized_numpy = vectorizer.NCBI_tokenizer_and_vectorizer(vocabulary,corpus_mentions,config['methods']['tokenizer'])

#padding for mentions
from keras.preprocessing.sequence import pad_sequences
logger.info('Old shape: {0}'.format(corpus_vectorized_numpy.shape))
corpus_vectorized_padded = pad_sequences(corpus_vectorized_numpy, padding='post')
logger.info('New shape: {0}'.format(corpus_vectorized_padded.shape))

#test MEDIC dictionary
logger.info('Loading dictionary...')
dictionary = load.load(config['terminology']['dict_file'],'MEDIC')
logger.info('Tokenizing and vectorizing dictionary terms...')
dictionary_tokenized, dictionary_vectorized = vectorizer.MEDIC_dict_tokenizer_and_vectorizer(dictionary,config['methods']['tokenizer'],vocabulary)

#candidate generation
import candidate_generation
logger.info('Generating candidates...')
dictionary_processed = candidate_generation.process_MEDIC_dict(dictionary_tokenized,config['methods']['candidate_generation'])
logger.info('Start generating candidates...')
generated_candidates = candidate_generation.generate_candidate(corpus_tokenized_mentions,dictionary_processed,config.getint('candidate','n'))
logger.info('Finished generating {0} candidates.'.format(len(corpus_tokenized_mentions)))

'''
#save candidates / load previously generated candidates
import tools
tools.output_generated_candidates(config['settings']['gencan_file'])
logger.info('Saving generated candidates...')

generated_candidates = tools.readin_generated_candidates(config['settings']['gencan_file'])
logger.info('Loading generated candidates...')
'''



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