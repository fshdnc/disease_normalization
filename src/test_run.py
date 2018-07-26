#!/usr/bin/env python3


import parse_NCBI_corpus
import json2obj
import json


'''converting dictionary object from parse_NCBI_disease_corpus generator to python objects
add the converted objects into a list 'NCBI_abstracts'
'''

NCBI_abstracts = []

for dict_abstract in parse_NCBI_corpus.parse_NCBI_disease_corpus("truncated_NCBI.txt"):
#trying to print just disease mention in a not brutal way
#ended up doing it the brutal way
#maybe change to self-defined class object?
        #print('dictionary:')
        #print(dict_abstract)
        #convert dictionary to json, then from json to python object
        print('converting dictionary to python object:')
        converted_object = json2obj.json2obj(json.dumps(dict_abstract))
        #print(converted_object)
        print(converted_object.sections)
        print('\n')

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


#the details of the resulting objects, how to use them?
### too much problem, try parsing into an object some other way


