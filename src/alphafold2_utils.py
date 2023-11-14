import pickle
import numpy as np
import os

class AlphaFold2Embed:
    """
    Embed generator for ESM and ProteinBERT
    """
    
    def __init__(self,
                 caldis_CA_loc='src/caldis_CA'):
        self.get_AA_dict()
        #self.seq = seq
        #self.loc = loc
        #self.model_name = model_name
        #self.pdb_rank = pdb_rank
        self.caldis_CA_loc = caldis_CA_loc
        #self.len = len(self.seq)

    def get_AA_dict(self):
        '''
        Return dictionary of amino acids
        '''
        AA = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY",
                "HIS", "ILE", "LYS", "LEU","MET", "ASN",
                "PRO", "GLN", "ARG", "SER", "THR", "VAL",
                "TRP", "TYR"]
        AA_abbr = [x for x in "ACDEFGHIKLMNPQRSTVWY"]
        self.AA_dict = dict(zip(AA, AA_abbr))

    def get_seq_from_pdb(self, pdb_loc):
        seq = ""
        current_pos = -1000
        with open(pdb_loc, "r") as f:
            lines = f.readlines()
        for line in lines:
            if line[0:4] == "ATOM" and int(line[22:26].strip()) != current_pos:
                aa_type = line[17:20].strip()
                seq += self.AA_dict[aa_type]
                current_pos = int(line[22:26].strip())
        return seq
    
    def get_distance_map(self, pdb_in):
        '''
        Extract Distnace maps
        '''
        # File names and paths setup
        path, filename = os.path.split(pdb_in)
        pdb_id = filename.replace('.pdb', '')
        maps_file = f'{path}/{pdb_id}.map'        
        # Get Protein seqs.
        PDB_seq = self.get_seq_from_pdb(pdb_in)
        
        # Run caldis_CA
        os.system(f"{self.caldis_CA_loc} {pdb_in}> {maps_file}")
        dis_map_seq, dis_map = self.process_distance_map(maps_file)
        
        # Save npy if everything is fine 
        if PDB_seq != dis_map_seq:
            raise Exception("PDB_seq & dismap_seq mismatch")
        else:
            return dis_map, dis_map_seq
        
    def process_distance_map(self, distance_map_file):
        '''
        Process Distnace maps
        '''
        with open(distance_map_file, "r") as f:
            lines = f.readlines()

        seq = lines[0].strip()
        length = len(seq)
        distance_map = np.zeros((length, length))

        if lines[1][0] == "#": # missed residues
            missed_idx = [int(x) for x in lines[1].split(":")[1].strip().split()] # 0-based
            lines = lines[2:]
        else:
            missed_idx = []
            lines = lines[1:]

        for i in range(0, len(lines)):
            record = lines[i].strip().split()
            for j in range(0, len(record)):
                distance_map[i + 1][j] = float(record[j])

        for idx in missed_idx:
            if idx > 0:
                distance_map[idx][idx - 1] = 3.8
            if idx > 1:
                distance_map[idx][idx - 2] = 5.4
            if idx < length - 1:
                distance_map[idx + 1][idx] = 3.8
            if idx < length - 2:
                distance_map[idx + 2][idx] = 5.4

        distance_map = distance_map + distance_map.T
        return seq, distance_map
    
    def generate_embeddig(self, pdb_loc):
        '''Get AlphaFold22 Reprersenation
        '''

        feas_file = pdb_loc.replace('.pdb', '.pkl')
        C, seq = self.get_distance_map(pdb_loc)
        with open(feas_file, "rb") as f:
            feas = pickle.loads(f.read())
            X = feas['representations']['single']
        return X, C, seq