# MSF-GZSSAR
This is the official implementation of the paper 'Multi-Semantic Fusion Model for Generalized Zero-Shot Skeleton-based Action Recognition', which has been accepted by ICIG 2023.

## Approach
![Alt pic](/figure/fig2.jpg)

## Requirements
<!-- ## Dependencies -->
* Python >= 3.8.13
* Torch >= 1.12.1
* Scikit-Learn

## Dataset: 
[NTU-60 & NTU-120](https://rose1.ntu.edu.sg/dataset/actionRecognition/)

## Data Preparation
To run the code, the skeleton features should be downloaded first. The skeleton features can be downloaded [here](https://drive.google.com/file/d/167xoVJQ684XU1uFhSKD6j9nAwHsnmEky/view). After downloading, rename the <code>synse_resources/ntu_results</code> folder to <code>sk_feats</code> and place it in the root directory of this repository.

## Running
<code>bash run60.sh</code> for the training&testing on NTU-60

<code>bash run120.sh</code> for the training&testing on NTU-120

## Details
* Seen-Unseen Splits for GZSSAR: 
  
  The seen-unseen splits for NTU-60 & NTU-120 (in <code>label_splits</code>) are the same with the SynSE, see [here](https://github.com/skelemoa/synse-zsl) for details.

* Skeleton Features:
  
  The skeleton features (in <code>sk_feats</code>) are extracted through [ShiftGCN](https://github.com/kchengiva/Shift-GCN).
  
* Text Features:
  
  3 different types of semantic information (i.e., class labels, action description and motion description) are provided in <code>sem_info</code>.The text features of the semantic information are provided in <code>text_feats</code>. They are extracted through the pre-trained ViT-B/32 which is the text encoder of [CLIP](https://github.com/openai/CLIP).
