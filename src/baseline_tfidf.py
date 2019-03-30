'''TF-IDF Baseline'''

import logging
import logging.config

import configparser as cp
import args

import pickle
import numpy as np

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


# concepts
concept_ids = [] # list of all concept ids
concept_all_ids = [] # list of (lists of all concept ids with alt IDs)
concept_names = [] # list of all names, same length as concept_ids
concept_map = {} # names as keys, ids as concepts

for k in dictionary.loaded.keys(): # keys should be in congruent order
    c_id = dictionary.loaded[k].DiseaseID
    a_ids = dictionary.loaded[k].AllDiseaseIDs
    
    if int(config['settings']['all_names']):
        for n in dictionary.loaded[k].AllNames:
            concept_ids.append(c_id)
            concept_all_ids.append(a_ids)
            concept_names.append(n)
            if n in concept_map: # one name corresponds to multiple concepts
                concept_map[n].append(c_id)
                # logger.warning('{0} already in the dictionary with id {1}'.format(n,concept_map[n]))
            else:
                concept_map[n] = [c_id]
    else:
        for n in dictionary.loaded[k].DiseaseName:
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
#corpus_train = sample.NewDataSet('training corpus')
#corpus_train.objects = load.load(config['corpus']['training_file'],'NCBI')

#corpus_dev = sample.NewDataSet('dev corpus')
#corpus_dev.objects = load.load(config['corpus']['development_file'],'NCBI')

corpus_test = sample.NewDataSet('test corpus')
corpus_test.objects = load.load('/home/lhchan/disease_normalization/data/NCBItestset_corpus.txt','NCBI')

corpus_dev = corpus_test

for corpus in [corpus_dev]:
    mention_ids = [] # list of all ids (gold standard for each mention)
    mention_names = [] # list of all names
    mention_all = [] # list of tuples (mention_text,gold,context,(start,end,docid))

    for abstract in corpus.objects:
        for section in abstract.sections: # title and abstract
            for mention in section.mentions:
                nor_ids = [sample._nor_id(one_id) for one_id in mention.id]
                mention_ids.append(nor_ids) # append list of ids, usually len(list)=1
                mention_names.append(mention.text)
                mention_all.append((mention.text,nor_ids,section.text,(mention.start,mention.end,abstract.docid)))

    # tokenization & vectorization of mentions
    #mention_tokenize = [nltk.word_tokenize(name) for name in mention_names]
    #mention_vectorize = np.array([[vocabulary.get(text,1) for text in mention] for mention in mention_tokenize])
    # mention_elmo = elmo_default([mention_names])

    corpus.ids = mention_ids
    corpus.names = mention_names
    corpus.all = mention_all
    # corpus.tokenize = mention_tokenize
    # corpus.vectorize = mention_vectorize
    # corpus.elmo = mention_elmo


#tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.snowball import PorterStemmer
stemmer = PorterStemmer()
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer(ngram_range=(1,3), analyzer='char')
# 'Pleuropneumonia, Contagious' becomes 'pleuropneumonia , contagi'
#concept_vectors = vectorizer.fit_transform([' '.join([stemmer.stem(w) for w in nltk.word_tokenize(c.lower())]) for c in concept.names])
#concept_vectors = vectorizer.fit_transform([' '.join([stemmer.stem(w) for w in nltk.word_tokenize(c.lower())]) for c in concept.names])

# Learn vocabulary and idf, return term-document matrix
concept_vectors = vectorizer.fit_transform(concept_names)

# Transform documents to document-term matrix.
mention_vectors = vectorizer.transform(corpus_dev.names)
#mention_vectors = vectorizer.transform([' '.join([stemmer.stem(w) for w in nltk.word_tokenize(c.lower())]) for c in corpus_dev.names])

sims = cosine_similarity(mention_vectors, concept_vectors)
prediction_indices = np.argmax(sims, axis=1)
predictions = np.array(concept.ids)[prediction_indices].tolist()


# calculate accuracy
correct = 0
#incorrect = 0
#incorrect_indices = []
for prediction, mention_gold in zip(predictions,corpus_dev.ids):
    if prediction == mention_gold[0] and len(mention_gold)==1:
        correct += 1
print('Accuracy:{0}'.format(correct/len(corpus_dev.names)))
    #[1] if men[0] in can and len(men)==1 else [0]

'''
tfidf
dev set
    without stemming, ngram(1,3): [0.7108] 0.7115628970775095 *2, 0.7090216010165185 *2, 0.7128335451080051
    stem concepts, mentions not stemmed, ngram(1,3): [0.6607] 0.6493011435832274, 0.6734434561626429, 0.6518424396442185, 0.6734434561626429, 0.6556543837357052
    stem concepts and mentions, ngram(1,3): [0.6681] 0.6747141041931385, 0.6709021601016518, 0.6734434561626429, 0.662007623888183, 0.6594663278271918
test set
    without stemming, ngram(1,3): [0.6246] 0.6197916666666666, 0.6291666666666667, 0.6145833333333334, 0.6239583333333333, 0.6354166666666666
    stem concepts, mentions not stemmed, ngram(1,3): [0.5979] 0.5916666666666667, 0.5875, 0.6020833333333333 *2, 0.60625
    stem concepts and mentions, ngram(1,3): [0.6142] 0.6239583333333333, 0.6041666666666666, 0.615625 *2, 0.6114583333333333
    without stemming, ngram(1,4): 0.6114583333333333
'''
