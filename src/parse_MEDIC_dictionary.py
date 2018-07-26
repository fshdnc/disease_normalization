#!/usr/bin7env python3
# coding:utf8

'''read in MEDIC terminology, terminology file from DNorm system
a lot of files in DNorm-0.0.7/data not looked into

read in MEDIC terminology into (namedtuples? /) dictionary?
read-in format:
	DiseaseName
	DiseaseID
	AltDiseaseIDs
	Definition
	ParentIDs
	TreeNumbers
	ParentTreeNumbers
	Synonyms'''

def parse_MEDIC_dictionary(filename):
	with open(filename,'r') as f:
		for line in f:
			if not line.startswith("#"):
				DiseaseName, DiseaseID, AltDiseaseIDs, Definition, _, _, _, Synonyms = line.split('\t')
				entry = {"DiseaseName": DiseaseName,
					"DiseaseID": DiseaseID,
					"AltDiseaseIDs": AltDiseaseIDs,
					"Definition": Definition,
					"Synonyms": Synonyms,
					}
				yield entry

