# SC-IQA: Few-label blind image quality assessment via samples chosen from new and existing scenes
This repository provides the official implementation of  
**Few-label blind image quality assessment via samples chosen from new and existing scenes**  ([Pattern Recognition, 2026](https://doi.org/10.1016/j.patcog.2025.112747)).

<img width="508.5" height="359.25" alt="image" src="https://github.com/user-attachments/assets/814c43a2-2478-44ba-8216-cad0ae983060" />

---

## Data Preparation:
To enable efficient computation in subsequent experiments, we store the images and corresponding labels of each dataset in `.mat` files. In addition, image features extracted in advance using the CLIP image encoder are also saved.
By running the provided `data_preprocess.py` scripts, all the above data preprocessing steps can be completed automatically.
## Sample Chosen:
The LIVEW database is used here as an example. After completing the above steps, the selected samples in both the new and existing scenes, as well as the aligned labels, can be obtained.
### 1. In the New Scene
The indices of samples selected in the new scene can be obtained by running the script `livew_NewScene_sample_chosen.py` in the ***Sample Chosen*** folder.
This script consists of three components: data distortion distribution, model prediction difficulty, and TOPSIS-based fusion.
### 2. In the Existing Scene
The samples selected for the existing scene can be obtained by running the script `livew_ExistingScene_sample_chosen.py` in the ***Sample Chosen*** folder.
This script consists of two main components: similarity-based sample selection and label alignment via nonlinear mapping.
## Training:
The training code can be available in the script `livew_training.py`. The alignment weights can be downloaded from: [weight](https://www.alipan.com/s/SsPs2NWyeSi). Please use 7-Zip application to unzip it.
## Testing:
The pre-trained models and alignment weights can be downloaded from: [Pre-trained models](https://www.alipan.com/s/SQKD92HtufH). Please download these files and put them in the same folder of code and then run `livew_testing.py`
## If you like this work, please cite:
@article{CHENG2026112747,
title = {Few-label blind image quality assessment via samples chosen from new and existing scenes},
journal = {Pattern Recognition},
volume = {172},
pages = {112747},
year = {2026},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2025.112747},
url = {https://www.sciencedirect.com/science/article/pii/S0031320325014104},
author = {Deqiang Cheng and Zihao Li and Tianshu Song and Qiqi Kou and Leida Li},
}
