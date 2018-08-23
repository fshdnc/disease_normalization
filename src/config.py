#!/usr/bin/env python3
#coding: utf8

import configparser as cp

#read config file
parser = cp.SafeConfigParser()
parser.read('defaults.cfg')

print parser.get('')

