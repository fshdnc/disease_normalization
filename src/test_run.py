#!/usr/bin/env python3


import parse_NCBI_corpus
import json2obj
import json


'''converting dictionary object from parse_NCBI_disease_corpus generator to python objects
add the converted objects into a list 'NCBI_abstracts'
'''

NCBI_abstracts = []

for dict_abstract in parse_NCBI_corpus.parse_NCBI_disease_corpus("gitig_truncated_NCBI.txt"):
        #convert dictionary to json, then from json to python object
        converted_object = json2obj.json2obj(json.dumps(dict_abstract))
        NCBI_abstracts.append(converted_object)

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
'''
print(NCBI_abstracts)

#the details of the resulting objects, how to use them?


