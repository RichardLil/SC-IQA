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

def train_loop(train_dataloader,model,loss_fn,optimizer,device,classname,epoch):
    model.train()
    s_loss = 0
    out_list = []
    label_list = []
    for idx, (X,Y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        X = X.to(device).float()
        torch.random.manual_seed(len(train_dataloader) * epoch + idx)
        rd_ps = torch.randint(20, (3,))
        X = X[:, :, rd_ps[0]:rd_ps[0] + 224, rd_ps[1]:rd_ps[1] + 224]
        if rd_ps[1] < 10:
            X = torch.flip(X, dims=[3])
        X /= 255
        X[:, 0] -= 0.485
        X[:, 1] -= 0.456
        X[:, 2] -= 0.406
        X[:, 0] /= 0.229
        X[:, 1] /= 0.224
        X[:, 2] /= 0.225
        Y = Y.to(device).float()
        Y -= 1
        Y /= 4

        text = torch.cat([clip.tokenize(f"{a} quality photo") for a in classname]).to(device)

        image_features = model.encode_image(X)
        text_features = model.encode_text(text)
        image_features = image_features/image_features.norm(dim=-1, keepdim=True)
        text_features = text_features/text_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        loss = loss_fn(probs[:,0].unsqueeze(1), Y)
        s_loss += loss.item()
        loss.backward()
        optimizer.step()

        out_list = np.concatenate((out_list, probs[:,0].detach().cpu().numpy()))
        label_list = np.concatenate((label_list, Y[:, 0].cpu().numpy()))

    print("train loss:{}".format(s_loss / len(train_dataloader)))
    plcc, _ = scipy.stats.pearsonr(out_list, label_list)
    print("[Train plcc]: {}".format(plcc))

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
    device = torch.device("cuda:0")

    model, preprocess = clip.load("ViT-B/32", device=device)
    model.float()

    model_weight_path = "koniq2livew_step0_best.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    ###################      New Date      ###################
    alldatas = sio.loadmat('livew_data_244.mat')
    X = alldatas['X']
    Y = alldatas['Y'].transpose(1, 0)
    Y = Y / 25 + 1
    del alldatas
    all_data = sio.loadmat('livew_NewScene_sample_chosen_result.mat')
    ind_train = all_data['train_ind'][:, 0]
    ind_test = all_data['test_ind'][:, 0]
    Xtest = X[ind_test]
    Ytest = Y[ind_test]
    X = X[ind_train]
    Y = Y[ind_train]

    ###################      Old Date      ###################
    # alldatas = sio.loadmat('koniq10k_data_244.mat')
    # X1 = alldatas['X']
    # alldatas1 = sio.loadmat('livew_ExistingScene_alignment.mat')
    # Y1 = alldatas1['koniq_Yalign']
    # all_data = sio.loadmat('livew_ExistingScene_sample_chosen.mat')
    # ind_train1 = all_data['train_ind'][:, 0]
    # ind_test1 = all_data['test_ind'][:, 0]
    # Xtest1 = X1[ind_test1]
    # Ytest1 = Y1[ind_test1]
    # X1 = X1[ind_train1]
    # Y1 = Y1[ind_train1]
    # del alldatas,alldatas1

    classname=['high','low']

    # 设置训练和测试的数据集内容
    training_data = MyData(X, Y)
    test_data = MyData(Xtest, Ytest)
    # training_data = MyData(X1, Y1)
    # test_data = MyData(Xtest1, Ytest1)
    # 加载训练和测试的数据集
    train_dataloader = DataLoader(training_data, batch_size=192, shuffle=True, pin_memory=True, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=256, shuffle=True, pin_memory=True, num_workers=0)
    # 开始梯度计算&优化器
    loss_fn = nn.MSELoss()
    learning_rate = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 保存模型地址
    # save_best_path = livew_training_step0_best.pth'
    # save_now_path = 'livew_training_step0_now.pth'

    save_best_path = 'livew_training_step1_best.pth'
    save_now_path = 'livew_training_step1_now.pth'

    best_test_plcc = 0
    best_test_srcc = 0

    epochs = 100

    consecutive_below_best=0

    for i in range(epochs):
        start_time = time.time()

        if i == 14:
            learning_rate = learning_rate * 0.3
        if i > 34 and learning_rate > 5e-8 and consecutive_below_best >= 2:
            learning_rate = learning_rate * 0.5
            consecutive_below_best = 0

        print(f"-----Epoch {i + 1}-----        learning rate：{learning_rate}")
        train_loop(train_dataloader, model, loss_fn, optimizer, device, classname,i)

        test_plcc, test_srcc = test_loop(test_dataloader, model, device, classname)
        if test_plcc > best_test_plcc:
            best_test_plcc = test_plcc
            best_plcc_epoch = i + 1
            torch.save(model.state_dict(), save_best_path)
            consecutive_below_best = 0
        else:
            consecutive_below_best += 1

        if test_srcc > best_test_srcc:
            best_test_srcc = test_srcc
            best_srcc_epoch = i + 1
        torch.save(model.state_dict(), save_now_path)

        end_time = time.time()
        all_time = end_time - start_time
        print(f"[Total Time]：{all_time}")
    print(f"best test plcc in {best_plcc_epoch} :{best_test_plcc}; best test srcc in {best_srcc_epoch} :{best_test_srcc}")


