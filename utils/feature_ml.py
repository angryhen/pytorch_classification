import torch
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.externals import joblib
import pandas as pd
import cv2

import os
from data.transform import create_transform
from configs.config import get_cfg_defaults
from data.dataloader import create_dataloader

from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import pickle
import numpy as np
from lib.use_model import choice_model


def get_config(configs_file):
    config = get_cfg_defaults()
    config.merge_from_file(configs_file)
    config.freeze()
    return config


def load_model():
    model = choice_model('efficientnet-b7', 15)
    if torch.cuda.is_available():
        model = model.cuda()
    ch = torch.load('../logs/2020-07-28 10:40:25/23.pth')
    model.load_state_dict(ch['state_dict'])
    model.eval()
    return model


NB_features = 2560


def save_feature(dataloader, feature_path, label_path):
    '''
    提取特征，保存为pkl文件
    '''
    model = load_model()
    print('..... Finished loading model! ......')
    if torch.cuda.is_available():
        model.cuda()
    nb_features = NB_features
    features = np.empty((len(dataloader), nb_features))
    labels = []
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(dataloader)):
            if torch.cuda.is_available():
                img = input.cuda(non_blocking=True)
                label = target.cuda(non_blocking=True)

            out = model.extract_features(img)
            out2 = nn.AdaptiveAvgPool2d(1)(out)
            feature = out2.view(out.size(1), -1).squeeze(1)
            features[i, :] = feature.cpu().numpy()
            label = label.cpu().numpy()
            labels.append(label)

    pickle.dump(features, open(feature_path, 'wb'))
    pickle.dump(labels, open(label_path, 'wb'))
    print('CNN features obtained and saved.')


def classifier_training(feature_path, label_path, save_path):
    '''
    训练分类器
    '''
    print('Pre-extracted features and labels found. Loading them ...')
    features = pickle.load(open(feature_path, 'rb'))
    labels = pickle.load(open(label_path, 'rb'))
    # classifier = SVC(C=0.5)
    classifier = MLPClassifier()
    # classifier = RandomForestClassifier(n_jobs=-1, criterion='entropy', n_estimators=70, min_samples_split=5)
    # classifier = KNeighborsClassifier(n_neighbors=4, n_jobs=-1)
    # classifier = ExtraTreesClassifier(n_jobs=-1,  n_estimators=100, criterion='gini', min_samples_split=10,
    #                        max_features=50, max_depth=40, min_samples_leaf=4)
    # classifier = GaussianNB()
    print(".... Start fitting this classifier ....")
    classifier.fit(features, labels)
    print("... Training process is down. Save the model ....")
    joblib.dump(classifier, save_path)
    print("... model is saved ...")


def classifier_pred(model_path, feature, id):
    '''
    得到测试集的预测结果
    '''
    features = pickle.load(open(feature, 'rb'))
    ids = pickle.load(open(id, 'rb'))
    print("... loading the model ...")
    classifier = joblib.load(model_path)
    print("... load model done and start predicting ...")
    predict = classifier.predict(features)
    # print(type(predict))
    # print(predict.shape)
    # print(ids)

    # print acc
    prediction = predict.tolist()
    pre = np.array(prediction)
    count = 0
    for i in range(len(ids)):
        if ids[i] == pre[i]:
            count += 1
    print(f'acc: {count / len(ids)}, total: {len(ids)}')

    # save
    submission = pd.DataFrame({"ID": ids, "Label": prediction})
    submission.to_csv('../' + 'svm_submission.csv', index=False, header=False)


if __name__ == "__main__":
    config = get_config('../configs/c15.yml')

    # #构建保存特征的文件夹
    feature_path = '../features/'
    os.makedirs(feature_path, exist_ok=True)

    # data -train
    train_data = pd.read_csv(config.train.dataset)
    train_loader = create_dataloader(config, train_data, 'train')
    # data -test
    test_data = pd.read_csv(config.test.dataset)
    test_loader = create_dataloader(config, test_data, 'test')

    #############################################################
    #### 保存训练集特征
    train_feature_path = feature_path + 'psdufeature_aug.pkl'
    train_label_path = feature_path + 'psdulabel.pkl'
    save_path = feature_path + 'psdusvm.m'

    test_feature_path = feature_path + 'testfeature_old.pkl'
    test_id_path = feature_path + 'testid_old.pkl'

    # save_feature(train_loader, train_feature_path, train_label_path)
    # save_feature(test_loader, test_feature_path, test_id_path)

    # train
    classifier_training(train_feature_path, train_label_path, save_path)

    ## #预测结果
    # save_path = feature_path + 'svm.m'
    classifier_pred(save_path, test_feature_path, test_id_path)
    classifier_pred(save_path, train_feature_path, train_label_path)
