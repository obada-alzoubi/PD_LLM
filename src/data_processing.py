import numpy as np


class PDDataProcessor:

    def __init__(self,
                 embedding_type='ESM',
                 seq_out_len=128,
                 scale_X=False,
                 scale_A=False,
                 thr_A=None,
                 auxiliary_data='data'):
        self.embedding_type = embedding_type
        self.seq_out_len = seq_out_len
        self.scale_X = scale_X
        self.scale_A = scale_A
        self.thr_A = thr_A
        self.auxiliary_data = auxiliary_data
        self.load_minmax()

    def process_data(self, X, A, seq, ind):
        '''
        X: A numpy array of shape (n, embed_dim)
        A: A numpy array of shape (n, n)
        seq: A sequence of amino acids with length of n
        ind: The index of the starting of embedding
        '''
        X = X[ind:ind+self.seq_out_len, :].astype(np.float32)
        if A is not None:
            A = A[ind:ind+self.seq_out_len, ind:ind+self.seq_out_len]
        seq = seq[ind:ind+self.seq_out_len]
        if self.scale_X:
            X = (X - self.minmax[self.embedding_type]["mins"]) / (
                (self.minmax[self.embedding_type]["maxs"] - self.minmax[self.embedding_type]["mins"]))
        # So don't normalize AF2 adj matrix even thoug we asked to do that. 
        if self.scale_A and A is not None:
            A = (A - A.min()) / (A.max() - A.min())
            np.fill_diagonal(A, 1)
        
        # Threshold A if necssary
        if self.thr_A is not None and A is not None:
            A = (A > self.thr_A).astype(np.float32)
        # 
        if A is None:
            return X[np.newaxis].astype(np.float32), [], seq
        return X[np.newaxis].astype(np.float32), A[np.newaxis].astype(np.float32), seq

    def load_minmax(self):
        """
        Load the minmax for each embedding
        """
        minmax = {}
        for embed in ["ESM", "AF2", "ProteinBERT"]:
            minmax[embed] = np.load(f"{self.auxiliary_data}/{embed}_minmax.npz")
        self.minmax = minmax