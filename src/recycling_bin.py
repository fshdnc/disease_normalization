'''
statistics of MEDIC
'''
# print all terms with more than one disease IDs
# result: 1483/9664
for k,v in dictionary.loaded.items():
    if len(v.AllDiseaseIDs)>1:
    	print(v.DiseaseID,'\t',v.AllDiseaseIDs)

count = 0
for k,v in dictionary.loaded.items():
    if len(v.AllDiseaseIDs)>1:
    	count += 1
print(count)

# put all disease names (no alternative names) into a list
DiseaseNames = [v.DiseaseName for k,v in dictionary.loaded.items()]
# put all disease names (including alternative names) into a list
AllNames = [v.AllNames for k,v in dictionary.loaded.items()]
# the flatten version of AllNames
flat_AllNames = [n for tu in AllNames for n in tu]

# count number of repeated names (no alternative names)
# result: 3
count = 0
for n in DiseaseNames:
	if flat_AllNames.count(n)>1:
		count += 1

# count number of repeated names (including alternative names)
# result: 229
count = 0
for n in flat_AllNames:
	if flat_AllNames.count(n)>1:
		count += 1

# result: 100
# mostly abbreviations
count_names = []
for n in flat_AllNames:
	count = flat_AllNames.count(n)
	if count>1:
		count_names.append((n,count))
len(set(count_names))

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


'''
predictions
code used when trying to figure out the format of the predictions
'''
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

'''
evaluations
code used for building custom callback
'''
correct = 0
for start, end, untok_mention in val_data.mentions:
	index_prediction = np.argmax(test_y[start:end],axis=0)
	index_gold = np.argmax(val_data.y[start:end],axis=0)
	if index_prediction == index_gold:
		correct += 1
logger.info('Accuracy: {0}, Correct: {1}, Total: {2}'.format(correct/len(val_data.mentions),correct,len(val_data.mentions)))
