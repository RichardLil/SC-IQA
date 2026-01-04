import scipy.io as sio
import numpy as np
import cv2
import torch
from clip import clip
import torch.utils.data
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
import os
import torchvision.transforms.functional as tf
from torch.cuda.amp import autocast as autocast


class MyData(Dataset):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return (self.images).shape[0]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def process_images_and_save_mat():

    all_data = sio.loadmat('E:\Database\LIVEW\LIVEW_mos')
    names = np.array(all_data['LIVEW_name'])
    imgs_lenth = len(names)
    labels = np.array(all_data['LIVEW_mos'])[:, 0]

    img_path = 'E:\Database\LIVEW\Images'

    imgs = np.zeros((imgs_lenth, 3, 224, 224), dtype=np.uint8)
    for i in np.arange(0, imgs_lenth):
        if i < 7:
            img = cv2.cvtColor(cv2.imread(img_path + '\\' + names[i, 0][0][28:], 1), cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(cv2.imread(img_path + '\\' + names[i][0][0][13:], 1), cv2.COLOR_BGR2RGB)

        imgs[i] = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)

    sio.savemat('livew_data_224.mat', {'X': imgs, 'Y': labels})
    return imgs, labels


def extract_clip_features(model, predict_loader, device):

    model.eval()
    image_features_list = []

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(predict_loader):
            data = data.to(device)
            data = tf.normalize(data / 255.0, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

            with autocast():
                image_features = model.encode_image(data)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features_list.append(image_features)

    image_features_list = torch.cat(image_features_list, dim=0)
    return image_features_list


if __name__ == '__main__':

    imgs, labels = process_images_and_save_mat()

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda")

    model, preprocess = clip.load("ViT-B/32", device=device)

    alldatas = sio.loadmat('livew_data_224.mat')
    X = alldatas['X']
    Y = alldatas['Y'].transpose(1, 0)

    del alldatas

    predict_dataset = MyData(X, Y)
    predict_loader = DataLoader(predict_dataset, batch_size=256 * 2, shuffle=False, pin_memory=True, num_workers=0)

    img_feature = extract_clip_features(model, predict_loader, device)
    img_feature = [feat.cpu().numpy() for feat in img_feature]

    sio.savemat('livew_imgfeature_224.mat', {'img_feature': img_feature})




