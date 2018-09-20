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

