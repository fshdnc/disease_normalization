'''exact match and lowercase baseline'''

import logging
import logging.config

import configparser as cp
import args

import vectorizer
import load
import sample

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
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level':'INFO',
            'formatter': 'standard',
            'class':'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': True
        }
    }
})


# MEDIC dictionary
dictionary = load.Terminology()
# dictionary of entries, key = canonical id, value = named tuple in the form of
#   MEDIC_ENTRY(DiseaseID='MESH:D005671', DiseaseName='Fused Teeth',
#   AllDiseaseIDs=('MESH:D005671',), AllNames=('Fused Teeth', 'Teeth, Fused')
dictionary.loaded = load.load(config['terminology']['dict_file'],'MEDIC')


concept_ids = [] # list of all concept ids
concept_all_ids = [] # list of (lists of all concept ids with alt IDs)
concept_names = [] # list of all names, same length as concept_ids
concept_map = {} # names as keys, ids as concepts

for k in dictionary.loaded.keys(): # keys should be in congruent order
    c_id = dictionary.loaded[k].DiseaseID
    a_ids = dictionary.loaded[k].AllDiseaseIDs
    
    for n in dictionary.loaded[k].AllNames:
        concept_ids.append(c_id)
        concept_all_ids.append(a_ids)
        concept_names.append(n)
        if n in concept_map: # one name corresponds to multiple concepts
            concept_map[n].append(c_id)
            # logger.warning('{0} already in the dictionary with id {1}'.format(n,concept_map[n]))
        else:
            concept_map[n] = [c_id]


# save the stuff to object
concept = sample.NewDataSet('concepts')
concept.ids = concept_ids
concept.all_ids = concept_all_ids
concept.names = concept_names
concept.map = concept_map


# corpus
corpus_test = sample.NewDataSet('test corpus')
corpus_test.objects = load.load('/home/lhchan/disease_normalization/data/NCBItestset_corpus.txt','NCBI')

# corpus_train = sample.NewDataSet('training corpus')
# corpus_train.objects = load.load(config['corpus']['training_file'],'NCBI')


for corpus in [corpus_test]:
    mention_ids = [] # list of all ids (gold standard for each mention)
    mention_names = [] # list of all names
    mention_all = [] # list of tuples (mention_text,gold,context,(start,end,docid))

    #import pdb;pdb.set_trace()
    for abstract in corpus.objects:
        for section in abstract.sections: # title and abstract
            for mention in section.mentions:
                nor_ids = [sample._nor_id(one_id) for one_id in mention.id]
                mention_ids.append(nor_ids) # append list of ids, usually len(list)=1
                mention_names.append(mention.text)
                mention_all.append((mention.text,nor_ids,section.text,(mention.start,mention.end,abstract.docid)))


	# save the stuff to object
    corpus.ids = mention_ids
    corpus.names = mention_names
    corpus.all = mention_all

# prediction
correct = 0
incorrect = 0
for mention_name, mention_gold in zip(corpus_test.names,corpus_test.ids):
	# string,     list
	prediction = []
	for concept_name, concept_id in zip(concept.names,concept.ids):
		# string,     string
		#if mention_name == concept_name:
		if mention_name.lower()==concept_name.lower():
			prediction.append(concept_id) # tuple or list?
	if set(prediction) != set(mention_gold):
		incorrect += 1
	else:
		correct += 1
print('Accuracy:{0}'.format(correct/len(corpus_test.names)))


# Test set:
# 	exact match: 142/960, acc: 0.147916667
# 	lowercase: 442/960, acc: 0.460416664

#correct = 0
incorrect = 0
incorrect_indices = []
for i, mention_name, mention_gold in zip(range(len(corpus_test.names)),corpus_test.names,corpus_test.ids):
    # string,     list
    prediction = []
    for concept_name, concept_id in zip(concept.names,concept.ids):
        # string,     string
        #if mention_name == concept_name:
        if mention_name.lower()==concept_name.lower():
            prediction.append(concept_id) # tuple or list?
    if set(prediction) != set(mention_gold):
        incorrect += 1
        incorrect_indices.append(i)
#print('Accuracy:{0}'.format(correct/len(corpus_test.names)))


# for printing out the test set and correct concepts
for i in incorrect_indices:
# for i,n in zip(corpus_test.ids,corpus_test.names):
    print('\n')
    try:
        print(corpus_test.names[i])
        print(dictionary.loaded[corpus_test.ids[i][0]])
    except KeyError:
        print(corpus_test.names[i])
        print('no dict item found')