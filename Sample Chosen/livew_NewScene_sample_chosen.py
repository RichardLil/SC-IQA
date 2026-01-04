import scipy.io as sio
import numpy as np
import torch
from clip import clip
from torch.utils.data import Dataset, DataLoader
import os
from torch.cuda.amp import autocast
from sklearn.cluster import KMeans
from scipy.stats import rankdata


# ================= Dataset =================
class MyData(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


# ================= Distortion Prediction =================
def predict_distortion(model, loader, classname, classname1, classname2, classname3, classname4, device):
    model.eval()
    bright, sharp, noisy, colorful, contrast = [], [], [], [], []

    with torch.no_grad():
        for data, _ in loader:
            data = np.squeeze(data.to(device).float(), 0)

            text = torch.cat([clip.tokenize(f"the photo features a {a} distribution of light and shadow") for a in classname]).to(device)
            text1 = torch.cat([clip.tokenize(f"{b} photo") for b in classname1]).to(device)
            text2 = torch.cat([clip.tokenize(f"the noise {c} the overall quality of this photo") for c in classname2]).to(device)
            text3 = torch.cat([clip.tokenize(f"{c} photo") for c in classname3]).to(device)
            text4 = torch.cat([clip.tokenize(f"{e} contrast photo") for e in classname4]).to(device)

            with autocast():
                image_features = data

                def _predict(t):
                    tf = model.encode_text(t)
                    tf /= tf.norm(dim=-1, keepdim=True)
                    return (100.0 * image_features @ tf.T).softmax(dim=-1)

                bright.append(_predict(text)[:, 0])
                sharp.append(_predict(text1)[:, 0])
                noisy.append(_predict(text2)[:, 0])
                colorful.append(_predict(text3)[:, 0])
                contrast.append(_predict(text4)[:, 0])

    return [torch.cat(x).cpu().numpy().reshape(-1, 1)
            for x in [bright, sharp, noisy, colorful, contrast]]


# ================= Dropout =================
def apply_dropout_to_output_layer(model, dropout_prob, i, s):
    def apply_mask(params):
        torch.random.manual_seed(s)
        mask = torch.rand(params.numel(), device=params.device) > dropout_prob
        return params * mask.view_as(params)

    for name, param in model.named_parameters():
        if i == 0 and any(k in name for k in [
            'resblocks.7.mlp', 'resblocks.8.mlp', 'resblocks.9.mlp']):
            param.data = apply_mask(param.data)
        if i == 1 and any(k in name for k in [
            'resblocks.10.mlp', 'resblocks.11.mlp']):
            param.data = apply_mask(param.data)


# ================= Quality Prediction =================
def predict_quality(model, loader, classname, device):
    model.eval()
    out = []

    with torch.no_grad():
        for data, _ in loader:
            data = np.squeeze(data.to(device).float(), 0)
            text = torch.cat([clip.tokenize(f"{a} quality photo") for a in classname]).to(device)

            with autocast():
                tf = model.encode_text(text)
                tf /= tf.norm(dim=-1, keepdim=True)
                out.append((100.0 * data @ tf.T).softmax(dim=-1))

    return torch.cat(out).cpu().numpy()


# ================= Utils =================
def min2max(x):
    return np.max(x) - x


def standard(x):
    norm = np.linalg.norm(x, axis=0)
    norm[norm == 0] = 1
    return x / norm


def topsis(z, w):
    zmax, zmin = z.max(axis=0), z.min(axis=0)
    S1 = np.sqrt(((z - zmax) ** 2 * w).sum(axis=1))
    S2 = np.sqrt(((z - zmin) ** 2 * w).sum(axis=1))
    return S2 / (S1 + S2)


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda")

    data = sio.loadmat('livew_imgfeature_224.mat')['img_feature']
    Y = sio.loadmat('livew_data_224.mat')['Y'].T

    loader = DataLoader(MyData(data, Y), batch_size=512, shuffle=False)

    # ---------- Distortion ----------
    model_dist, _ = clip.load("ViT-B/32", device=device)
    distortions = predict_distortion(
        model_dist, loader,
        ['good', 'poor'],
        ['sharp', 'blurry'],
        ['enhances', 'reduces'],
        ['colorful', 'dull'],
        ['good', 'poor'],
        device
    )
    alldist = np.hstack(distortions)

    # ---------- KMeans ----------
    estimator = KMeans(n_clusters=7, random_state=82)
    labels = estimator.fit_predict(alldist)
    centroids = estimator.cluster_centers_
    distances = np.linalg.norm(alldist - centroids[labels], axis=1, keepdims=True)

    # ---------- Uncertainty ----------
    model_q1, _ = clip.load("ViT-B/32", device=device)
    model_q2, _ = clip.load("ViT-B/32", device=device)

    high = predict_quality(model_q1, loader, ['high', 'low'], device)[:, 0]
    apply_dropout_to_output_layer(model_q1, 0.5, 0, 93)
    high1 = predict_quality(model_q1, loader, ['high', 'low'], device)[:, 0]
    apply_dropout_to_output_layer(model_q2, 0.5, 1, 93)
    high2 = predict_quality(model_q2, loader, ['high', 'low'], device)[:, 0]

    std = np.std(np.c_[high, high1, high2], axis=1, keepdims=True)

    # ---------- TOPSIS ----------
    data_topsis = standard(np.hstack((min2max(distances), std)))
    score = topsis(data_topsis, np.array([0.45, 0.55]))
    rank = rankdata(-score, method='min')

    split = int(len(rank) * 0.25)
    order = np.argsort(rank)

    sio.savemat('livew_NewScene_sample_chosen_result.mat', {
        'train_ind': order[:split].reshape(-1, 1),
        'test_ind': order[split:].reshape(-1, 1)

    })
