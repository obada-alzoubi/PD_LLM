# Efficient Protein-DNA Binding Prediction Using Large Language Models
The GitHub repository dedicated to the research on “Efficient Protein-DNA Binding Prediction Using Large Language Models”, as outlined in the previously discussed abstract, serves as a digital resource for scientists and researchers in the field. This repository hosts implementation details and examples of the innovative approach using large language models (LLMs) for extracting embeddings from protein sequences. For those keen on replicating or building upon the research, detailed documentation guides users through the utilization of Evolutionary Scale Modeling (ESM)-v2, ProteinBERT (PB), and the AlphaFold2 embeddings. Additionally, the repository contains pre-trained models, especially the graph attention networks that achieved an AUC of around 90%, thus offering a ready-to-use solution for those aiming to study Protein-DNA interactions. 

## Protein Represenation Generation
### ProteinBERT Embeddings 
ProteinBERT can be installed from this repository: [ProteinBERT](https://github.com/nadavbra/protein_bert) </br>
```
from pyfaidx import Fasta
from src.proteinBERT_utils import ProteinBERTEmbed 
seqs = Fasta('data/3u58_A.fa')
seq = str(seqs[0][:128]) # Get the first 128 Amino acid residues.
print(seq)
PB_seq = ProteinBERTEmbed()
X, _ = PB_seq.generate_embeddig(seq)
print(X)

```
### ESM-v2
Similarly, ESM-v2 model can be installed from this repository: [ESM-v2](https://github.com/facebookresearch/esm)

```
from pyfaidx import Fasta
from src.ESM_utils import ESMEmbed
seqs = Fasta('data/3u58_A.fa')
seq = str(seqs[0][:128]) # Get the first 128 Amino acid residues.
print(seq)
ESM_seq_rpres = ESMEmbed(representation_layer=33)
X, C = ESM_seq_rpres.generate_embeddig(seq)
print(X)
print(C)
```
In the previous, code `representation_layer=33` set the index of the layer to extrac represenation from. 
### AlphaFold2 
AlphaFold2 can be installed and setup using the following repository [AlphaFold2](https://github.com/google-deepmind/alphafold) </br>

Since AlphaFold2 is computationally expensive, running AlphaFold2 on the provided examples was done offline. </br>

Assuming the results of AlphaFold2 were stored in `loc`:
```
dir
|--id.pdb
|--id.pkl
```
The following piece of code extracts the protein presentation (single representation) in addition to calculating distances among amino acid residues from the predicted 3D structure. 


```
from src.alphafold2_utils import AlphaFold2Embed
pdb_loc = 'data/AF2/demo/1bssa/1bss.A.pdb'
AF2_seq_rpres = AlphaFold2Embed()
X, C, seq = AF2_seq_rpres.generate_embeddig(pdb_loc)
print(X)
print(C)

```
The code will extract the single represenation only and will run calculate distnace maps among CA atoms using `'src/caldis_CA'` script. Several other postprocessing steps are perfromed to ensure accurate alignment among amino acid residues and distnace maps. 

## Embedding Preprocessing for Prediction
Once embeddings are generated from either ProteinBERT, ESM-v2, AlpahFold2 or a combination of the previous, data are ready for postprocessing before feeding running PD models. Additionally, ESM-v2 and AlphaFold2 can generate contact/distance maps. Thus,   We may have to process distance/contact maps for some PD  models like GAT one. </br>
 The goal of postprocessing is to feed PD models with the right sequence and scale embeddings and contact/distance maps if needed.  In our setup, we used a sequence length of `n=128` in all PD models to allow for fair comparisons among different architectures. es. 

```
from src.data_processing import PDDataProcessor
embedding_type = 'ESM' # ESM or ProteinBERT
ind = 0 # index of starting location. 
ESM_DP = PDDataProcessor(embedding_type=embedding_type, seq_out_len=128,  scale_X = False, scale_A=True, thr_A=None)
X_model, C_model, Seq_model = ESM_DP.process_data(X, C, seq, ind)
print(X_model)
print(C_model)
print(Seq_model)

```

## Running Portein-DNA (PD) Prediction  
Having processed the embedding and potential contact/distance maps, the next piece of code will take the input data and generate a PD score per each amino acid sequence.  We trained a wide range of  PD models; however, we provide two trained models: A Graph Attention (GAT) model trained on ESM-v2 embedding and contact maps from ESM-v2 and another GAT trained on AF2 single representation and distance maps from  AF2-C(AF2). 

```
from src.prediction_utils import EsmCEsmPDModel
ESM_PD_Predictor = EsmCEsmPDModel('models/ESM_C_ESM/saved_model')
prediction = ESM_PD_Predictor.predict(X, C)
print(prediction)

```
The repo cotains two models:
- **ESM-C(ESM)**: Graph Attention Tensorflow model trained using ESM-v2 embedding with contact maps from ESM-v2. 
- **AF2-C(AF2)**: Graph Attention Tensorflow model trained using AF2 embedding with contact maps from AF2. 

## Demo Example 
A complete example for predicting the PD scores from the first `n=128` amino acid residues of chain A from *3U58* protein. 

```
from pyfaidx import Fasta
import numpy as np
from src.ESM_utils import ESMEmbed
from src.data_processing import PDDataProcessor
from src.prediction_utils import EsmCEsmPDModel

embedding_type = 'ESM' # ESM or ProteinBERT
model_loc = 'models/ESM_C_ESM/saved_model'
seq = Fasta('data/3u58_A.fa')

seq = str(seq[0][:128]) # Get the first 128 Amino acid residues.
print(seq)
# Generate Embedding
ESM_seq_rpres = ESMEmbed(representation_layer=33)
X, C = ESM_seq_rpres.generate_embeddig(seq)
# Process Embedding for to make it ready for prediction 
ESM_DP = PDDataProcessor(embedding_type=embedding_type, seq_out_len=128,  scale_X = False, scale_A=False, thr_A=None)
X_model, C_model, Seq_model = ESM_DP.process_data(X, C, seq, 0)
# Load ESM-C(ESM) model
ESM_PD_Predictor = EsmCEsmPDModel(model_loc)
# Predict Protein-DNA interactions 
prediction = ESM_PD_Predictor.predict(X_model, C_model)
print(prediction)
# Load sequence labels
seq_labels = Fasta('data/3u58_A.labels')
seq_labels = str(seq_labels[0][:128])
seq_labels = [int(l) for l in seq_labels]
seq_labels = np.array(seq_labels)
np.where(seq_labels==1)[0]
print(np.where(prediction>=0.5)[0])

```
A complete set of tutorials along with visualization can be found under [notebooks](./notebooks)
## Reference

## Maintainer 
For any questions, please send us email at:
oalzoubi@omegatx.com
