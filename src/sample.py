#!/usr/bin/env python3

'''class objects for dataset and samples'''

class DataSet:
    def __init__(self):
        self.objects = None
        self.mentions = None
        self.tokenized_mentions = None
        self.vectorized_numpy_mentions = None
        self.padded = None

class Sample:
    def __init__(self):
        self.dummy = None
