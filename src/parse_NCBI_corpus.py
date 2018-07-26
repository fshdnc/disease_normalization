
#!/usr/bin/env python3
# coding: utf8

'''@Arguments
Filename: "NCBItrainset_corpus.txt", "NCBIdevelopset_corpus.txt", and "NCBItestset_corpus.txt"


read file and convert to text
then text converted into dictionary
read dict into text and label like NER
'''

def parse_NCBI_disease_corpus(filename):
    '''parse the NCBI disease corpus'''
    with open(filename,"r",encoding="ascii") as file:
        for entry in _split_file(file):
            yield _parse_entry(entry)

def _split_file(file_object):
    '''break file objects into entries'''
    entry=[]
    for line in file_object:
        line = line.rstrip()
        if line:
            entry.append(line)
        elif entry:
            yield entry
            entry.clear()
    # do not miss the final entry
    if entry:
        yield entry

def _parse_id(id):
    '''for  composite mention and multiple concept'''
    ##character does not exist still returns a list
    #if not (('|' in id) or ('+' in id)):
    #    cache=[]
    #    return cache.append(id)
    #elif "+" in id:
    if "+" in id:
        return id.split("+")
    else:
        return id.split("|")


def _parse_entry(cache_entry):
    '''parse entry into desired format
    cache_entry is a list where each item is a line of the entry'''
    title_line=cache_entry[0].split('|')
    docid=title_line[0] #id is in the first position
    title=max(title_line,key=len)
    abstract=max(cache_entry[1].split('|'),key=len)
    abstract_offset=len(title)+1
    title_mentions=[]
    abstract_mentions=[]
    for mention in cache_entry[2:]: #the mentions are documented from the third line
        cache_mention=mention.split('\t')
        cache_mention[:] = [item for item in cache_mention if item != ''] #some lists contain empty elements because of the ugly format :P
        if int(cache_mention[1])<abstract_offset:
            cache_dict={'start':int(cache_mention[1]),
                        'end':int(cache_mention[2]),
                        'text':cache_mention[3],
                        'type':cache_mention[4],
                        'id':_parse_id(cache_mention[5])}
            title_mentions.append(cache_dict)
            assert cache_dict['id'] != None #'id' was none for the first entry in the old version
        else:
            cache_dict={'start':int(cache_mention[1])-abstract_offset,
                        'end':int(cache_mention[2])-abstract_offset,
                        'text':cache_mention[3],
                        'type':cache_mention[4],
                        'id':_parse_id(cache_mention[5])}
            abstract_mentions.append(cache_dict)
    sections=[{'text':title,'offset':0,'mentions':title_mentions},
              {'text':abstract,'offset':abstract_offset,'mentions':abstract_mentions}]
    return {'docid':docid,'sections':sections}


#test_data=parse_NCBI_disease_corpus("NCBItestset_corpus.txt")
#print(len(test_data))
