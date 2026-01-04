# SC-IQA: Few-label blind image quality assessment via samples chosen from new and existing scenes
This is the source code for [Few-label blind image quality assessment via samples chosen from new and existing scenes](https://doi.org/10.1016/j.patcog.2025.112747).
<img width="678" height="479" alt="image" src="https://github.com/user-attachments/assets/814c43a2-2478-44ba-8216-cad0ae983060" />
## For Data Preparetion:
To enable efficient computation in subsequent experiments, we store the images and corresponding labels of each dataset in '.mat' files. In addition, image features extracted in advance using the CLIP image encoder are also saved.
By running the provided 'data_preprocess.py' scripts, all the above data preprocessing steps can be completed automatically.
## For Sample Chosen:
### 1. In the New Scene
