# SC-IQA
This repository provides the official implementation of **Few-label blind image quality assessment via samples chosen from new and existing scenes**  ([Pattern Recognition, 2026](https://doi.org/10.1016/j.patcog.2025.112747)).

<img width="508.5" height="359.25" alt="image" src="https://github.com/user-attachments/assets/814c43a2-2478-44ba-8216-cad0ae983060" />

---

## Environment
- Pytorch: 1.8.1
- CUDA: 10.2
- Python: 3.7

---

## Data Preparation:
To enable efficient computation in subsequent experiments, the images and corresponding labels of each dataset are stored in `.mat` files. In addition, image features extracted in advance using the CLIP image encoder are also saved.  
By running the provided `data_preprocess.py` script, all the above data preprocessing steps can be completed automatically.


---

## Sample Chosen:
The LIVEW database is used here as an example. After completing the following steps, the selected samples in both the new and existing scenes, as well as the aligned labels, will be obtained and saved in `.mat` format.

### 1. New Scene
The indices of samples selected in the new scene can be obtained by running the script `livew_NewScene_sample_chosen.py` in the ***Sample Chosen*** folder.
This script consists of three components: data distortion distribution, model prediction difficulty, and TOPSIS-based fusion.

### 2. Existing Scene

The samples selected for the existing scene can be obtained by running the script `livew_ExistingScene_sample_chosen.py` in the ***Sample Chosen*** folder.  
This script consists of two main components: similarity-based sample selection and label alignment via nonlinear mapping.

---

## Training:
The training code is provided in the script `livew_training.py`.  
The alignment weights can be downloaded from: [Alignment weights](https://www.alipan.com/s/SsPs2NWyeSi). Please use the 7-Zip application to unzip the downloaded file.

---

## Testing:
The pre-trained models can be downloaded from: [Pre-trained models](https://www.alipan.com/s/SQKD92HtufH). Please place the downloaded files in the same directory as the code and then run `livew_testing.py`.

---

## Citation
If you find this work useful, please cite:

```bibtex
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
