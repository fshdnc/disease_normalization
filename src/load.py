#!/usr/bin/env python3


import parse_NCBI_corpus
import json2obj
import json

def load(filename,type):
        '''type: 'NCBI' or 'MEDIC'
        '''
        if type=='NCBI':
                return _load_NCBI_discorp(filename)
        elif type=='MEDIC':
                return _load_MEDIC(filename)


def _load_NCBI_discorp(filename):
        '''converting dictionary object from parse_NCBI_disease_corpus generator to python objects
           add the converted objects into a list 'NCBI_abstracts'

           filenames:
           gitig_truncated_NCBI.txt

           NCBIdevelopset_corpus.txt
           NCBItestset_corpus.txt
           NCBItrainset_corpus.txt
        '''
        NCBI_abstracts = []

        for dict_abstract in parse_NCBI_corpus.parse_NCBI_disease_corpus(filename):
                #convert dictionary to json, then from json to python object
                converted_object = json2obj.json2obj(json.dumps(dict_abstract))
                NCBI_abstracts.append(converted_object)
        return NCBI_abstracts

        '''sanity check
                print('mentions in a section:')
                cache_pot = []
                for piece_of_text in converted_object.sections:
                        print(converted_object.docid)
                        cache_pot.append(converted_object.docid)
                        #print(piece_of_text)
                        if piece_of_text.mentions:
                	            for mention in piece_of_text.mentions:
                                        print(mention.id)
                                        cache_pot.append(mention.id)
                                        print(mention.text)
                                        cache_pot.append(mention.text)
                                        print('\n')

                print('comparsion: access from original dictionary')
                cache_dict = []
                for dict_section in dict_abstract['sections']:
                        print(dict_abstract['docid'])
                        cache_dict.append(dict_abstract['docid'])
                        for dict_mention in dict_section['mentions']:
                                print(dict_mention['id'])
                                cache_dict.append(dict_mention['id'])
                                print(dict_mention['text'])
                                cache_dict.append(dict_mention['text'])
                        print('\n')
                assert cache_pot == cache_dict
                print('\n\n')

        print(NCBI_abstracts)
        '''


#from gitig_MEDIC.py
import parse_MEDIC_dictionary

def _load_MEDIC(filename):

        '''MEDIC terms from generator to tuples (saves memory)
           saved in a dictionary (DiseaseID as key)'''

        MEDIC_dict = {}

        for DiseaseID, entry in parse_MEDIC_dictionary.parse_MEDIC_dictionary(filename):
                #print(entry,'\n')
                assert DiseaseID not in MEDIC_dict.keys()
                MEDIC_dict[DiseaseID] = entry
        return MEDIC_dict


'''object for dictionary'''
class Terminology:
    def __init__(self):
        self.loaded = None
        self.tokenized = None
        self.vectorized = None
        self.processed = None
        self.no_cangen_tokenized = None
        self.no_cangen_vectorized = None