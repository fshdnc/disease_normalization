import vectorizer
import load
import sample
import configparser as cp
import args
import logging
import numpy as np

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
'''
#incorporate this into the rest somehow
def main():
    import logging.congig
    logging.config.fileConfig('/path/to/logging.conf')

if __name__ == '__main__':
	main
'''
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
corpus = sample.DataSet()
corpus.objects = load.load(config['corpus']['corpus_file'],'NCBI')
#list of mentions
#each mention has a docid and a sections, which contains title and abstract
corpus.mentions = [mention.text for obj in corpus.objects for part in obj.sections for mention in part.mentions]
logger.info('Tokenizing and vectorizing mentions...')
corpus.tokenized_mentions, corpus.vectorized_numpy_mentions = vectorizer.NCBI_tokenizer_and_vectorizer(vocabulary,corpus.mentions,config['methods']['tokenizer'])

#padding for mentions
from keras.preprocessing.sequence import pad_sequences
logger.info('Old shape: {0}'.format(corpus.vectorized_numpy_mentions.shape))
corpus.padded = pad_sequences(corpus.vectorized_numpy_mentions, padding='post')
#format of corpus.padded: numpy, mentions, padded
logger.info('New shape: {0}'.format(corpus.padded.shape))

#test MEDIC dictionary
logger.info('Loading dictionary...')
dictionary = load.Terminology()
dictionary.loaded = load.load(config['terminology']['dict_file'],'MEDIC')

logger.info('Tokenizing and vectorizing dictionary terms...')
dictionary.tokenized, dictionary.vectorized = vectorizer.MEDIC_dict_tokenizer_and_vectorizer(dictionary.loaded,config['methods']['tokenizer'],vocabulary)
'''
logger.info('Tokenizing dictionary terms...')
dictionary.tokenized = vectorizer.MEDIC_dict_tokenizer(dictionary.loaded,config['methods']['tokenizer'],vocabulary)
'''

#candidate generation
import candidate_generation
'''
logger.info('Generating candidates...')
dictionary.processed = candidate_generation.process_MEDIC_dict(dictionary.tokenized,config['methods']['candidate_generation'])
logger.info('Start generating candidates...')
training_data = sample.Sample()
logger.warning('Using only first 100 mentions!')
#training_data.generated = candidate_generation.generate_candidate(corpus.tokenized_mentions,dictionary.processed,dictionary.tokenized,dictionary.vectorized,config.getint('candidate','n'))
training_data.generated = candidate_generation.generate_candidate(corpus.tokenized_mentions[:100],dictionary.processed,dictionary.tokenized,dictionary.vectorized,config.getint('candidate','n'))
logger.info('Finished generating {0} candidates.'.format(len(training_data.generated)))
'''

'''
#save candidates / load previously generated candidates
import pickle
with open(config['settings']['gencan_file'],'wb') as f:
    pickle.dump(training_data.generated,f)

logger.info('Saving generated candidates...')
'''
import pickle
training_data = sample.Sample()
training_data.generated = pickle.load(open(config['settings']['gencan_file'],'rb'))
logger.info('Loading generated candidates...')


#formatting generated candidates
logger.info('Formatting candidates...')
#function parameters need updating
#modify format_candidates to remove 100
sample.format_candidates(training_data,corpus,dictionary.vectorized)

#debug previous line!






'''
for mention_vector, candidates in zip(corpus_vectorized_padded[:100],generated_candidates):
	

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