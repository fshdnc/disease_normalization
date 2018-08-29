#!/usr/bin/env python3

def output_generated_candidates(filename):
    with open(filename,'w',encoding='utf-8') as f:
        for item in generated_candidates:
            f.write("%s\n" % item)
def readin_generated_candidates(filename):
    with open(filename,'r',encoding='utf-8') as f:
        generated_candidates = []
        while True:
            mention = f.readline()
            if not mention:
                break
            l = mention[2:-3].split('), (')
            candidates = []
            for j in l:
                candidate = j.split(', ')
                candidates.append((candidate[0][1:-1],float(candidate[1])))
            generated_candidates.append(candidates)
    return generated_candidates
