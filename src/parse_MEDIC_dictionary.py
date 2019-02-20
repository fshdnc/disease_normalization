#!/usr/bin7env python3
# coding:utf8

from collections import namedtuple

'''read in MEDIC terminology, terminology file from DNorm system
a lot of files in DNorm-0.0.7/data not looked into

read in MEDIC terminology into namedtuples
read-in format:
	DiseaseName
	DiseaseID
	AltDiseaseIDs
	Definition (usually none)
	ParentIDs
	TreeNumbers
	ParentTreeNumbers
	Synonyms

returned format:
namedtuple(
	DiseaseID
	DiseaseName
	AllDiseaseIDs: DiseaseID + AltDiseaseIDs
	AllNames: DiseaseName + Synonyms

'''

MEDIC_ENTRY = namedtuple('MEDIC_ENTRY','DiseaseID DiseaseName AllDiseaseIDs AllNames')

#namedtuple: https://stackoverflow.com/questions/2970608/what-are-named-tuples-in-pythons
def parse_MEDIC_dictionary(filename):
	with open(filename,'r') as f:
		for line in f:
			if not line.startswith("#"):
				DiseaseName, DiseaseID, AltDiseaseIDs, Def, _, _, _, Synonyms = line.strip('\n').split('\t')
				AllDiseaseIDs = tuple([DiseaseID]+AltDiseaseIDs.split('|')) if AltDiseaseIDs else tuple([DiseaseID])
				AllNames = tuple([DiseaseName]+Synonyms.split('|')) if Synonyms else tuple([DiseaseName])
				entry = MEDIC_ENTRY(DiseaseID,DiseaseName,AllDiseaseIDs,AllNames)
				yield DiseaseID, entry

#dunno what will happend if no altID or syn
#some AllNames tuples have comma at the end but does not seem to affect the functionality