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
vector_model, vocabulary, inversed_vocabulary = vectorizer.prepare_embedding_vocab('~/old-disease-normalization/data/embeddings/wvec_50_haodi-li-et-al.bin', binary = True, limit = 50000)
pretrained = vectorizer.load_pretrained_word_embeddings(vocabulary, vector_model)

#test MEDIC dictionary
logger.info('Loading dictionary...')
dictionary = load.Terminology()
dictionary.loaded = load.load(config['terminology']['dict_file'],'MEDIC')
#import pdb; pdb.set_trace()
logger.info('Tokenizing and vectorizing dictionary terms...')
if int(config['candidate']['use']):
	dictionary.tokenized, dictionary.vectorized = vectorizer.MEDIC_dict_tokenizer_and_vectorizer(dictionary.loaded,config['methods']['tokenizer'],vocabulary)
	'''
	logger.info('Tokenizing dictionary terms...')
	dictionary.tokenized = vectorizer.MEDIC_dict_tokenizer(dictionary.loaded,config['methods']['tokenizer'],vocabulary)
	'''
else:
	dictionary.no_cangen_tokenized = vectorizer.MEDIC_dict_tokenizer_no_cangen(dictionary.loaded,config['methods']['tokenizer'])
	dictionary.no_cangen_vectorized = vectorizer.MEDIC_dict_vectorizer_no_cangen(dictionary.no_cangen_tokenized,vocabulary)

#NCBI corpus
##test NCBI corpus, the mentions are not separated by abstract
#list of objects, '/home/lhchan/disease-normalization/data/ncbi-disease/NCBItestset_corpus.txt'
logger.info('Loading NCBI training corpus...')
corpus_train = sample.DataSet()
corpus_train.objects = load.load(config['corpus']['training_file'],'NCBI')
#list of mentions
#each mention has a docid and a sections, which contains title and abstract
corpus_train.mentions = [mention.text for obj in corpus_train.objects for part in obj.sections for mention in part.mentions]
logger.info('Tokenizing and vectorizing training mentions...')
corpus_train.tokenized_mentions, corpus_train.vectorized_numpy_mentions = vectorizer.NCBI_tokenizer_and_vectorizer(vocabulary,corpus_train.mentions,config['methods']['tokenizer'])
logger.info('Formatting training mention ids...')
corpus_train.mention_ids = [sample.canonical_id_list(mention.id,dictionary.loaded) for obj in corpus_train.objects for part in obj.sections for mention in part.mentions]

# development set
logger.info('Loading NCBI development corpus...')
corpus_dev = sample.DataSet()
corpus_dev.objects = load.load(config['corpus']['development_file'],'NCBI')
#list of mentions
#each mention has a docid and a sections, which contains title and abstract
corpus_dev.mentions = [mention.text for obj in corpus_dev.objects for part in obj.sections for mention in part.mentions]
logger.info('Tokenizing and vectorizing test mentions...')
corpus_dev.tokenized_mentions, corpus_dev.vectorized_numpy_mentions = vectorizer.NCBI_tokenizer_and_vectorizer(vocabulary,corpus_dev.mentions,config['methods']['tokenizer'])
logger.info('Formatting test mention ids...')
corpus_dev.mention_ids = [sample.canonical_id_list(mention.id,dictionary.loaded) for obj in corpus_dev.objects for part in obj.sections for mention in part.mentions]


#padding for mentions
from keras.preprocessing.sequence import pad_sequences
for corpus in [corpus_train,corpus_dev]:
	logger.info('Old shape: {0}'.format(corpus.vectorized_numpy_mentions.shape))
	corpus.padded = pad_sequences(corpus.vectorized_numpy_mentions, padding='post', maxlen=int(config['embedding']['length']))
	#format of corpus.padded: numpy, mentions, padded
	logger.info('New shape: {0}'.format(corpus.padded.shape))


if int(config['candidate']['use']): # if can_gen is used
#training set, candidate generation
	logger.info('Using candidate generation...')
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
	training_data.generated = pickle.load(open(config['settings']['gencan_file_train'],'rb'))
	#training_data.generated = pickle.load(open('gitig_generated_candidates_all.txt','rb'))
	logger.info('Loading generated candidates...')

	#formatting generated candidates
	logger.info('Formatting candidates...')
	sample.format_candidates(training_data,corpus_train,dictionary.vectorized)

	#format y
	logger.info('Checking candidates...')
	sample.check_candidates(training_data,corpus_train.mention_ids)
	
# validation set
	logger.info('Candidate generation for validation set...')
	if not dictionary.processed:
		dictionary.processed = candidate_generation.process_MEDIC_dict(dictionary.tokenized,config['methods']['candidate_generation'])
	val_data = sample.Sample()
	'''
	val_data.generated = candidate_generation.generate_candidate(corpus.tokenized_mentions,dictionary.processed,dictionary.tokenized,dictionary.vectorized,config.getint('candidate','n'))
	logger.info('Finished generating {0} candidates for the validation set.'.format(len(training_data.generated)))
	# save generated candidates for the development set
	import pickle
	with open(config['settings']['gencan_file_dev'],'wb') as f:
	    pickle.dump(val_data.generated,f)
	logger.info('Saving generated candidates for tge validation set...')
	'''
	import pickle
	val_data.generated = pickle.load(open(config['settings']['gencan_file_dev'],'rb'))
	sample.format_candidates(val_data,corpus_dev,dictionary.vectorized)
	#import pdb; pdb.set_trace()
	sample.check_candidates(val_data,corpus_dev.mention_ids)
else: #try not to use candidates
	logger.info('Not using candidate generation...')
	if not int(config['settings']['use_saved_data_no_cangen']):
		training_data = sample.Sample()
		val_data = sample.Sample()

		import candidate_generation
		can_list = candidate_generation.Candidates()
		# assign lists to can_list
		candidate_generation.all_candidates(can_list,dictionary.no_cangen_tokenized,dictionary.no_cangen_vectorized)
		can_list.canonical = [sample._canonical(key,dictionary.loaded.keys(),dictionary.loaded) for key in can_list.keys]
		
		logger.info('Formatting CNN inputs...')
		logger.warning('Not formatting Data.generated.')
		for corpus, data in zip([corpus_train,corpus_dev],[training_data,val_data]):
			data.x = sample.no_cangen_format_x(corpus.padded,can_list.vectorized)
			data.mentions = sample.no_cangen_format_mentions(corpus.mentions,len(can_list.vectorized))
			data.y = sample.no_cangen_format_y(can_list.canonical,corpus.mention_ids)
			assert len(data.x[0]) == len(data.y)
		#import pdb; pdb.set_trace()
		# debug len(data.x[0])!=len(data.y)
		#debug can_list.canonical==can_list.keys True

		import pickle
		data = [[training_data.x,training_data.y,training_data.mentions],[val_data.x,val_data.y,val_data.mentions]]
		with open(config['settings']['data_no_cangen'],'wb') as f:
			pickle.dump(data,f,protocol=4)
		logger.info('Training and validation inputs saved.')
	else:
		logger.info('Loading pre-saved formatted data for CNN input.')
		logger.warning('Not creating candidate_generation.Candidates() object.')
		import pickle
		data = pickle.load(open(config['settings']['data_no_cangen'],'rb'))
		training_data = sample.Sample()
		val_data = sample.Sample()
		sample.load_no_cangen_data(data,training_data,val_data)


if not int(config['model']['use_saved_model']):	   # train new model
	import cnn, model_tools
	cnn.print_input(training_data)
	model = cnn.build_model(config,training_data,vocabulary,pretrained)
	hist = model.fit(training_data.x, training_data.y, epochs=1, batch_size=100)
	#hist = model.fit(training_data.x, training_data.y, epochs=10, batch_size=100)
	# WARNING (theano.tensor.blas): We did not found a dynamic library into the library_dir of the library we use for blas. If you use ATLAS, make sure to compile it with dynamics library.
	logger.info('Saving newly trained model...')
	model_tools.save_model(model,config['model']['path_model_architecture'],config['model']['path_model_weights'])
else:
	from cnn import semantic_similarity_layer
	import cnn, model_tools
	model = model_tools.load_model(config['model']['path_model_architecture'],config['model']['path_model_weights'],{'semantic_similarity_layer': semantic_similarity_layer})
	model.compile(optimizer='adadelta',loss='binary_crossentropy')

# predictions
if int(config['settings']['predict']):
	logger.info('Making predictions...')
	test_y = model.predict(val_data.x)
	if int(config['settings']['save_prediction']):
		logger.info('Saving predictions to {0}'.format(config['model']['path_saved_predictions']))
		model_tools.save_predictions(config['model']['path_saved_predictions'],test_y) #(filename,predictions)

if int(config['settings']['load_prediction']):
	logger.info('Loading predictions from {0}'.format(config['settings']['path_saved_predictions']))
	import pickle
	test_y = pickle.load(open(config['settings']['path_saved_predictions'],'rb'))

# evaluations
correct = 0
for start, end, untok_mention in val_data.mentions:
	index_prediction = np.argmax(test_y[start:end],axis=0)
	index_gold = np.argmax(val_data.y[start:end],axis=0)
	if index_prediction == index_gold:
		correct += 1
logger.info('Accuracy: {0}, Correct: {1}, Total: {2}'.format(correct/len(val_data.mentions),correct,len(val_data.mentions)))



# 	import pdb; pdb.set_trace()

#>>> dictionary.loaded['MESH:D014314'].AllDiseaseIDs
#('MESH:D014314', 'MESH:D000782', 'MESH:D058674')

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