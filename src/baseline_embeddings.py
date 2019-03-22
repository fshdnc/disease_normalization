'''Word2Vec Baseline
python3 baseline_embeddings.py path_to_embedding'''

import logging
import logging.config

import configparser as cp
#import args
import sys

import pickle
import numpy as np

import vectorizer
import load
import sample

#configurations
config = cp.ConfigParser(strict=False)
config.read('defaults.cfg')

#argparser
#args = args.get_args()
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


# word embedding
from gensim.models import KeyedVectors

def prepare_embedding_vocab(filename, binary = True, limit = 1000000):
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

def load_pretrained_word_embeddings(vocab,embedding_model):
    """vocab: vocabulary from data vectorizer
       embedding_model: model loaded with gensim"""
    pretrained_embeddings = np.random.uniform(low=-0.05, high=0.05, size=(len(vocab)-1,embedding_model.vectors.shape[1]))
    pretrained_embeddings = np.vstack((np.zeros(shape=(1,embedding_model.vectors.shape[1])), pretrained_embeddings))
    found=0
    for word,idx in vocab.items():
        if word in embedding_model.vocab:
            pretrained_embeddings[idx]=embedding_model.get_vector(word)
            found+=1           
    logger.info("Found pretrained vectors for {found} words.".format(found=found))
    return pretrained_embeddings

def emb_baseline(emb_path):
    #vector_model, vocabulary, inversed_vocabulary = prepare_embedding_vocab('/home/lenz/disease-normalization/data/embeddings/wvec_200_win-30_chiu-et-al.bin')
    vector_model, vocabulary, inversed_vocabulary = prepare_embedding_vocab(emb_path, binary = True)
    pretrained = load_pretrained_word_embeddings(vocabulary, vector_model)


    # MEDIC dictionary
    dictionary = load.Terminology()
    # dictionary of entries, key = canonical id, value = named tuple in the form of
    #   MEDIC_ENTRY(DiseaseID='MESH:D005671', DiseaseName='Fused Teeth',
    #   AllDiseaseIDs=('MESH:D005671',), AllNames=('Fused Teeth', 'Teeth, Fused')
    dictionary.loaded = load.load(config['terminology']['dict_file'],'MEDIC')

    import vectorizer
    dictionary.no_cangen_tokenized = vectorizer.MEDIC_dict_tokenizer_no_cangen(dictionary.loaded,config['methods']['tokenizer'])
    dictionary.no_cangen_vectorized = vectorizer.MEDIC_dict_vectorizer_no_cangen(dictionary.no_cangen_tokenized,vocabulary)


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

    #concept_vectorize = np.array([dictionary.no_cangen_vectorized[k] for k in concept.ids])


    # corpus
    #corpus_train = sample.NewDataSet('training corpus')
    #corpus_train.objects = load.load(config['corpus']['training_file'],'NCBI')

    corpus_dev = sample.NewDataSet('dev corpus')
    corpus_dev.objects = load.load(config['corpus']['development_file'],'NCBI')

    #corpus_test = sample.NewDataSet('test corpus')
    #corpus_test.objects = load.load('/home/lhchan/disease_normalization/data/NCBItestset_corpus.txt','NCBI')
    #corpus_dev=corpus_test

    for corpus in [corpus_dev]:
        mention_ids = [] # list of all ids (gold standard for each mention)
        mention_names = [] # list of all names
        mention_all = [] # list of tuples (mention_text,gold,context,(start,end,docid))

        #sth wrong here that sometimes throw an error
        #import pdb;pdb.set_trace()
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


    # vector representations
    import nltk
    mention_embeddings = []
    for mention in corpus.names:
        tokenized = nltk.word_tokenize(mention.lower())
        index = [vocabulary.get(token,1) for token in tokenized]
        #emb = np.mean(np.array([pretrained[i] for i in index]), axis=0)
        emb = np.sum(np.array([pretrained[i] for i in index]), axis=0)
        mention_embeddings.append(emb)
    mention_embeddings = np.array(mention_embeddings)

    concept_embeddings = []
    for mention in concept.names:
        tokenized = nltk.word_tokenize(mention.lower())
        index = [vocabulary.get(token,1) for token in tokenized]
        #emb = np.mean(np.array([pretrained[i] for i in index]), axis=0)
        emb = np.sum(np.array([pretrained[i] for i in index]), axis=0)
        concept_embeddings.append(emb)
    concept_embeddings = np.array(concept_embeddings)




    '''
    from vectorizer_elmo import elmo_default
    # chunk the concepts down since the list is too big
    concept_chunk = [concept.names[i:i + 5000] for i in range(0, len(concept.names), 5000)]
    concept.elmo = []
    for chunk in concept_chunk:
        [elmo_chunk] = [c for c in elmo_default([chunk])]
        concept.elmo.append(elmo_chunk)
    [concept.elmo] = [chunk for chunk in elmo_default([concept_chunk])]

    #with open('gitig_concept_elmo.pickle','wb') as f:
    #    pickle.dump(concept.elmo,f,protocol=4)

    #concept.elmo = pickle.load(open('gitig_concept_elmo.pickle','rb'))

    concept.elmo =  np.array([item for sublist in concept.elmo for item in sublist])
    [corpus_dev.elmo] = [chunk for chunk in elmo_default([corpus_dev.names])]
    '''

    concept_emb = concept_embeddings #concept.elmo
    mention_emb = mention_embeddings #corpus_dev.elmo

    from sklearn.preprocessing import normalize
    nor_concepts = normalize(concept_emb)
    nor_corpus_dev = normalize(mention_emb)

    dot_product_matrix = np.dot(nor_corpus_dev,np.transpose(nor_concepts))
    prediction_indices = np.argmax(dot_product_matrix,axis=1)
    predictions = np.array(concept.ids)[prediction_indices].tolist()


    correct = 0
    #incorrect = 0
    #incorrect_indices = []
    for prediction, mention_gold in zip(predictions,corpus_dev.ids):
        if prediction == mention_gold[0] and len(mention_gold)==1:
            correct += 1
    print('Accuracy:{0}'.format(correct/len(corpus_dev.names)))
        #[1] if men[0] in can and len(men)==1 else [0]


if __name__ == '__main__':
    emb_baseline(sys.argv[1])
    #normalize(sys.argv[1], sys.argv[2])