#!/usr/bin/env python3
#coding: utf8

import argparse as ap

def get_args():
	parser = ap.ArgumentParser(description = '')
	parser.add_argument(
		'-t','--train',action='store_true',
		help='train a ranking model')
	#parser.add_argument(
	#	'-m','--model',metavar='PATH')
	args=parser.parse_args()
	return args
