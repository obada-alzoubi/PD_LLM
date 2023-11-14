from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
import numpy as np


class ProteinBERTEmbed:
    """
    Embed generator for ESM and ProteinBERT
    """
    
    def __init__(self):
        pass
        
    def generate_embeddig(self, seq, batch_size=1):
        '''Generate Protien BERT Reprersenation
        '''
        seq_len = len(seq)
        pretrained_model_generator, input_encoder = load_pretrained_model()
        model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(seq_len+2))
        x_reps = input_encoder.encode_X([seq], seq_len)
        x, _ = model.predict(x_reps, batch_size=batch_size)
        x = x[:, 1:-1, :]
        return np.squeeze(x), []