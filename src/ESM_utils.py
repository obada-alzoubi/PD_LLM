import torch
import esm
import numpy as np


class ESMEmbed:
    """
    Embed generator for ESM and ProteinBERT
    """
    
    def __init__(self,
                 representation_layer=33):
        self.representation_layer = representation_layer


    def generate_embeddig(self, seq, name="protein1"):
        '''
        Generate ESM Emeddings
        '''
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic results
        data = [(name, seq)]
        _, _, batch_tokens = batch_converter(data)
        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[self.representation_layer],
                            return_contacts=True)
        token_representations = results["representations"][self.representation_layer]
        # Remove start and end tokens
        x = token_representations[0][1:-1, :]
        # get contact maps
        contacts = results["contacts"].numpy()
        # return numpy array
        return np.squeeze(x.numpy()), np.squeeze(contacts)