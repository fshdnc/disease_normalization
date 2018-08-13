from subword_nmt.apply_bpe import BPE

def _load_bpe_model(model):
    '''model name:
       '/home/lhchan/glove/selftrained_bpe_model/abstract_lowered_5000model'
    '''
    with open (model,encoding='utf8') as f:
        bpe = BPE(f)
    return bpe

def _load_bpe_vocab(vocab_file):
    '''
    Takes a vocab file, return the vocab in a list
    vocab file name:
       '/home/lhchan/glove/selftrained_bpe_model/vocab_lowered_5000_100.txt'
    '''
    vocabs = []
    with open(vocab_file,encoding='utf8') as f:
        for line in f:
            vocab, _ = line.split(' ')
            vocabs.append(vocab)
    return vocabs

def check_coverage(mentions,vocab_file,model,lowercase=True):
    '''
       mentions: list of untokenized mentions
       vocab: vocab file
    '''
    vocabs = _load_bpe_vocab(vocab_file)
    bpe = load_bpe_model(model)
    denominator = 0
    numerator = 0
    mentions_cased = [mention.lower() if lowercase == True else mention for mention in corpus_mentions]
    for mention in mentions_cased:
        snippets = bpe.segment(mention).split(' ')
        denominator = denominator + len(snippets)
        for snippet in snippets:
            if snippet in vocabs:
                numerator += 1
    print('File name:',vocab_file)
    print('Coverage of vocab:',float(numerator)/denominator,'\n')

files = [('vocab_lowered_1000_50.txt','abstract_lowered_1000model',True),
         ('vocab_lowered_1000_100_3.txt','abstract_lowered_1000model',True),
         ('vocab_3000_50.txt','abstract3000model',False),
         ('vocab_lowered_3000_50.txt','abstract_lowered_3000model',True),
         ('vocab_lowered_3000_100_3.txt','abstract_lowered_3000model',True),
         ('vocab_lowered_5000_50.txt','abstract_lowered_5000model',True),
         ('vocab_lowered_5000_100_3.txt','abstract_lowered_5000model',True),
         ('vocab_lowered_7500_50.txt','abstract_lowered_7500model',True),
         ('vocab_lowered_7500_100_3.txt','abstract_lowered_7500model',True),
         ('vocab_10000_50.txt','abstract10000model',False),
         ('vocab_lowered_10000_50.txt','abstract_lowered_10000model',True),
         ('vocab_lowered_10000_100_3.txt','abstract_lowered_10000model',True),
         ('vocab_lowered_zero_10000_50.txt','abstract_lowered_zero_10000model',True),
         ('vocab_20000_50.txt','abstract20000model',False),
         ]

for file in files:
    vocab_name = '/home/lhchan/glove/selftrained_bpe_model/'+file[0]
    model_name = '/home/lhchan/glove/selftrained_bpe_model/'+file[1]
    check_coverage(corpus_mentions,vocab_name,model_name,file[2])
