import os
import shutil
import time
import scipy.stats
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from clip import clip

class MyData(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = torch.FloatTensor(labels)
    def __len__(self):
        return (self.images).shape[0]
    def __getitem__(self, index):
        return torch.from_numpy(self.images[index]),self.labels[index]


def test_loop(test_dataloader,model,device,classname):
    model.eval()
    with torch.no_grad():
        out_list = []
        label_list = []
        for X, Y in test_dataloader:
            X = X.float().to(device)
            X = X[:, :, 10:10 + 224, 10:10 + 224]

            X /= 255
            X[:, 0] -= 0.485
            X[:, 1] -= 0.456
            X[:, 2] -= 0.406
            X[:, 0] /= 0.229
            X[:, 1] /= 0.224
            X[:, 2] /= 0.225
            Y = Y.float().numpy()
            Y -= 1
            Y /= 4

            text = torch.cat([clip.tokenize(f"{a} quality photo") for a in classname]).to(device)

            image_features = model.encode_image(X)
            text_features = model.encode_text(text)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            out_list = np.concatenate((out_list, probs[:, 0].cpu().numpy()))
            label_list = np.concatenate((label_list, Y[:, 0]))
        plcc, _ = scipy.stats.pearsonr(out_list, label_list)
        srcc, _ = scipy.stats.spearmanr(out_list, label_list)
        print("[Test plcc]: {}".format(plcc))
        print("[Test srcc]: {}".format(srcc))

    return plcc,srcc

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda")
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.float()

    model_weight_path = "livew_training_step1_best.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    ###################      LIVEW      ###################
    alldatas = sio.loadmat('livew_data_244.mat')
    X = alldatas['X']
    Y = alldatas['Y'].transpose(1, 0)
    Y = Y / 25 + 1
    del alldatas
    all_data = sio.loadmat('livew_NewScene_sample_chosen_result.mat')
    ind_test = all_data['test_ind'][:, 0]
    Xtest = X[ind_test]
    Ytest = Y[ind_test]

    classname=['high','low']

    test_data = MyData(Xtest, Ytest)
    test_dataloader = DataLoader(test_data, batch_size=256, shuffle=True, pin_memory=True, num_workers=0)
    test_plcc, test_srcc = test_loop(test_dataloader, model, device, classname)

    print(f"test plcc:{test_plcc}; test srcc :{test_srcc}")