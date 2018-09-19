#!/usr/bin/env python3

import logging
logger = logging.getLogger(__name__)

def output_generated_candidates(filename,generated_candidates):
    with open(filename,'w',encoding='utf-8') as f:
        for item in generated_candidates:
            f.write("%s\n" % item)
def readin_generated_candidates(filename):
    with open(filename,'r',encoding='utf-8') as f:
        generated_candidates = []
        while True:
            mention = f.readline()
            if not mention:
                break
            l = mention[2:-3].split('), (')
            candidates = []
            for j in l:
                candidate = j.split(', ')
                candidates.append((candidate[0][1:-1],float(candidate[1])))
            generated_candidates.append(candidates)
    return generated_candidates

'''
def shelve_working_env(filename):
    import shelve
    logger.info('Saving working environment variables...')
    my_shelf = shelve.open(filename,'n') #'n' for new
    import pdb; pdb.set_trace()
    #broken, dir() gives only variables from this module
    for key in dir():
        try:
            my_shelf[key] = globals()[key]
        except TypeError:
            logger.error('__builtins__, my_shelf, and imported modules cannot be shelve-ed')
            logger.error('ERROR shelving: {0}'.format(key))
    logger.info('Working environment variables saved to {0}'.format(filename))
    my_shelf.close()
'''