#!/usr/bin/env python3

import numpy
import vectorizer


def tok_whole_mention():
	'''
	
	'''
	pass #dummy

corpus_intermediate = [[mention.text,part.text,part.offset,mention.start,mention.end] for obj in corpus.objects for part in obj.sections for mention in part.mentions]
corpus.mentions = [mention.text for obj in corpus.objects for part in obj.sections for mention in part.mentions]
