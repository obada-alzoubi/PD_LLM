
def read_FASTA(fname):
    '''
    Read a fasta file
    '''
    fil = open(fname, "rt")
    lins = fil.readlines()
    fil.close()
    id = lins[0].split(' ')[0][1:]
    seq = lins[1]
    return id, seq