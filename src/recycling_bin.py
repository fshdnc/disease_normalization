'''
file: run-py
old version of saving generated candidates
'''
import tools
tools.output_generated_candidates(config['settings']['gencan_file'],training_data.generated)
logger.info('Saving generated candidates...')

import tools
training_data = sample.Sample()
training_data.generated = tools.readin_generated_candidates(config['settings']['gencan_file'])
logger.info('Loading generated candidates...')

'''
file: run.py
tryiny to pickle objects
problems:
   (1) seem to only pickle one object
   (2) have unpicklable objects
'''

try:
    import cPickle as pickle
except:
    import pickle

with open('pickled','wb') as f:
	for o in dir():
		pickle.dump(o,f)

with open('pickled','rb') as f:
	x = pickle.load(f)


'''
file: run.py
trying to save working environment
problem: pickling error
'''
tools.shelve_working_env('gitig_cureent_working_env')


'''
file: run.py
trying to bulk-pickle stuff
'''
import dill
dill.dump_session('gitig_cureent_working_env')

import dill
#load the session
dill.load_session('gitig_cureent_working_env')

'''
file: run.py
trying to look at the statistics of the mentions
'''
dummy_for_corpus_mention_ids = [mention.id for obj in corpus.objects for part in obj.sections for mention in part.mentions]

MESH=[]
OMIM=[]
multiple = []
for id_list in dummy_for_corpus_mention_ids:
	if len(id_list)==1:
		if 'OMIM' not in id_list[0]:
			MESH.extend(id_list)
		else:
			OMIM.extend(id_list)
	else:
		multiple.append(id_list)

MESH2=[]
OMIM2=[]
for id_list in dummy_for_corpus_mention_ids:
	for ans in id_list:
		if 'OMIM' not in ans:
			MESH2.extend(id_list)
		else:
			OMIM2.extend(id_list)
'''
file: run.py
count the number of IDs that are not keys
'''
count = 0
for id_list in dummy_for_corpus_mention_ids:
    for ans in id_list:
        if ans not in dictionary.loaded.keys():
            count+=1
#count = 149, need to disambiguate the IDs