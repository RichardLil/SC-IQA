import os
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
from clip import clip


# ================= Model =================
class MyModel(nn.Module):
    def __init__(self, model1, head):
        super().__init__()
        self.model = model1
        self.head = head

    def forward(self, X, classname):
        text = torch.cat([clip.tokenize(f"{a} quality photo") for a in classname]).to(X.device)

        image_features = self.model.encode_image(X)
        text_features = self.model.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = probs[:, 0:1]

        op1 = (4 / (torch.exp((self.head[0, 0] - probs) / torch.abs(self.head[0, 1])) + 1)) + 1
        op2 = (4 / (torch.exp((self.head[1, 0] - probs) / torch.abs(self.head[1, 1])) + 1)) + 1
        return op1, op2


# ================= Sample Selection =================
def select_existing_scene_samples():
    livew_imgfeature = sio.loadmat('livew_imgfeature_224.mat')['img_feature']

    ind_train = sio.loadmat('livew_NewScene_sample_chosen_result.mat')['train_ind'][:, 0]
    livew_imgfeature = livew_imgfeature[ind_train]

    koniq10k_imgfeature = sio.loadmat('../ADATABASE/00_koniq_imgfeature_224.mat')['img_feature']

    livew_imgfeature = livew_imgfeature.reshape(livew_imgfeature.shape[0], -1)
    koniq10k_imgfeature = koniq10k_imgfeature.reshape(koniq10k_imgfeature.shape[0], -1)

    sim = livew_imgfeature @ koniq10k_imgfeature.T

    top = np.argsort(-sim, axis=1)[:, :3]
    train_ind = np.unique(top.reshape(-1), axis=0)

    all_idx = np.arange(10073)
    test_ind = np.setdiff1d(all_idx, train_ind)

    return sim, train_ind.reshape(-1, 1), test_ind.reshape(-1, 1)


# ================= Label Alignment =================
def align_koniq_labels(device):
    model1, _ = clip.load("ViT-B/32", device=device)
    model = MyModel(model1, head=nn.Parameter(torch.randn((2, 2))))

    model.load_state_dict(torch.load(
        "livew_koniq10k_align_model_weight.pth",
        map_location=device
    ))

    prs = model.head.data.cpu().numpy()

    # Y = sio.loadmat('koniq10k_data_244.mat')['Y'].T
    Y = sio.loadmat('../../training/dataset/koniq10k_244_forclip.mat')['Y'].T

    Yr = prs[1, 0] - abs(prs[1, 1]) * np.log((5 - Y) / (Y - 1))
    Yalign = (4 / (np.exp((prs[0, 0] - Yr) / np.abs(prs[0, 1])) + 1)) + 1

    return Yalign


# ================= Main =================
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda")

    # ---- sample chosen ----
    sim, train_ind, test_ind = select_existing_scene_samples()
    sio.savemat('livew_ExistingScene_sample_chosen.mat', {
        'sim': sim,
        'train_ind': train_ind,
        'test_ind': test_ind
    })

    # ---- label alignment ----
    Yalign = align_koniq_labels(device)
    sio.savemat('livew_ExistingScene_alignment.mat', {
        'koniq_Yalign': Yalign
    })
