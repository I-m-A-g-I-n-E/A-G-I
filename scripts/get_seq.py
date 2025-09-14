d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

def three_to_one(seq):
    return "".join([d[seq[i:i+3]] for i in range(0, len(seq), 3)])

with open('af_pdbs/AF-P69905-F1-model_v4.pdb', 'r') as f:
    seq = ''
    for line in f:
        if line.startswith('SEQRES'):
            seq += ''.join(line.split()[4:])
    print(three_to_one(seq))
