# -*- coding: utf-8 -*-
# OMP_NUM_THREADS=1 python python_file_change.py
import pandas as pd
import cv2
import os
from scipy import ndimage
from scipy.spatial import distance
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import tqdm
import numpy as np
base_path = './dataset'
train_path = os.path.join(base_path, 'traindata')
test_path = os.path.join(base_path, 'testdata')
import pandas as pd
import pickle    
import glob
import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy.cluster.vq import vq
train_paths=glob.glob('./dataset/traindata/*/*.jpg')
test_paths=glob.glob('./dataset/testdata/*.jpg')
import math
df_data=pd.read_csv(os.path.join(base_path, 'Label2Names.csv'), header=None)

CODEBOOKSIZE=4096
codebook_path='./codeword_kmeans_geometric_4096.pkl'

import sys
import random
device_gpu = torch.device('cuda')
device_cpu = torch.device('cpu')
####################################################################
########################### dataseet load ##########################
####################################################################
def get_DSIFT_feature(img):
    sift=cv2.xfeatures2d.SIFT_create()
    w, h = img.shape
    keypoints=[cv2.KeyPoint(i,j,8) 
                for i in range(0, h, 8)  
                for j in range(0, w, 8)]
    kp, des = sift.compute(img, keypoints)
    return kp, des

def get_DSIFT_keypoint(img):
    sift=cv2.xfeatures2d.SIFT_create()
    w, h = img.shape
    keypoints=[cv2.KeyPoint(i,j,8) 
                for i in range(0, h, 8)  
                for j in range(0, w, 8)]
    return keypoints


def histogram(des, codebook):
    codeword, _ = vq(des, codebook)
    his, _ = np.histogram(codeword, bins=list(range(n_cluster+1)))

    return his

def make_kp_arr(kp_list):
    kp_array=[]
    for kp in kp_list:
        #image_kp=[]
        for xy in kp:
            point=[]
            for info in xy.pt:
                point.append(info)
            #image_kp.append(point)
            kp_array.append(point)
    return kp_array

kp_list=[]
des_list=[]
trainLabels=[]
if os.path.exists('./train_feature.pkl'):
    for train_path in tqdm(train_paths):
        image=cv2.imread(train_path)
        image=cv2.resize(image,(256,256))
        image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp=get_DSIFT_keypoint(image) # kp[0].pt 
        kp_list.append(kp)
    kp_list=make_kp_arr(kp_list)
    print('load train_feature.pkl')
    with open('./train_feature.pkl', 'rb') as tf:
        des_list=pickle.load(tf)
    print('load train_Label_list.pkl')
    with open('./train_Label_list.pkl', 'rb') as tf:
        trainLabels=pickle.load(tf)
else:
    for train_path in tqdm(train_paths):
        image=cv2.imread(train_path)
        image=cv2.resize(image,(256,256))
        image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, des=get_DSIFT_feature(image)
        kp_list.append(kp)
        des_list.append(des)

        classes=train_path.split('/')[-2]
        if classes=="BACKGROUND_Google":
              labelind=102
        else:
            labelind=(df_data.index[df_data[1]==classes]+1).tolist()[0]
        trainLabels.append(labelind)
    kp_list=make_kp_arr(kp_list)
    print('make train_feature.pkl')
    with open('./train_feature.pkl', 'wb') as tf:
        pickle.dump(des_list,tf)
    print('make train_Label_list.pkl')
    with open('./train_Label_list.pkl', 'wb') as tf:
        pickle.dump(trainLabels,tf)

test_kp_list=[]
test_de_list=[]
if not os.path.exists('./test_feature.pkl'):
    for test_path in tqdm(test_paths):
        image=cv2.imread(test_path)
        image=cv2.resize(image,(256,256))
        image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, des=get_DSIFT_feature(image) # kp[0].pt 
        test_kp_list.append(kp)
        test_de_list.append(des)
    test_kp_list=make_kp_arr(test_kp_list)
    print('make test_feature.pkl')
    with open('./test_feature.pkl', 'wb') as tf:
        pickle.dump(test_de_list,tf)
    print('make test_path_list.pkl')
    with open('./test_path_list.pkl', 'wb') as tf:
        pickle.dump(test_paths,tf)
else:
    for test_path in tqdm(test_paths):
        image=cv2.imread(test_path)
        image=cv2.resize(image,(256,256))
        image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp=get_DSIFT_keypoint(image) # kp[0].pt 
        test_kp_list.append(kp)
    test_kp_list=make_kp_arr(test_kp_list)
    print('load test_feature.pkl')
    with open('./test_feature.pkl', 'rb') as tf:
        test_de_list=pickle.load(tf)
    print('load test_path_list.pkl')
    with open('./test_path_list.pkl', 'rb') as tf:
        test_paths=pickle.load(tf)
####################################################################
########################### dataseet load ##########################
####################################################################
# test_paths test_de_list test_kp_list
# kp_list des_list trainLabels

####################################################################
####################### geometric extension  #######################
####################################################################
#test_kp_list test_de_list
#kp_list des_list

def normalize_extension(x_train_des,x_train_kp):
    #import pdb;pdb.set_trace()
    img_num=np.array(x_train_des).shape[0]
    des_num=np.array(x_train_des).shape[1] # np.array(x_train_des).shape (3060, 1024, 128)
    x_train_des=np.transpose(x_train_des,(2,0,1)) # x_train_des.shape (128, 3060, 1024)
    x_train_des=x_train_des.reshape(x_train_des.shape[0],x_train_des.shape[1]*x_train_des.shape[2]) # (128, 3133440=1024*3060)
    ###########normalize
    sqrt=np.sqrt(np.square(x_train_des).sum(0))
    sqrt[sqrt==0]=1e-5
    nor=1./sqrt
    nor=nor.reshape(1,-1)
    x_train_des=x_train_des*nor

    ###########################################################
    mean=x_train_des.mean(1).reshape(-1,1)
    x_train_des=x_train_des-mean
    ########################################################
    sqrt=np.sqrt(np.square(x_train_des).sum(0))
    sqrt[sqrt==0]=1e-5
    nor=1./sqrt
    nor=nor.reshape(1,-1)
    x_train_des=x_train_des*nor
    #########################################################
    ################geomatric_extension
    
    x_train_kp=x_train_kp-256/2
    x_train_kp=x_train_kp/256
    x_train_kp=np.transpose(x_train_kp,(1,0))
    x_train_des=np.concatenate((x_train_des,x_train_kp))
    #########################################################
    sqrt=np.sqrt(np.square(x_train_des).sum(0))
    sqrt[sqrt==0]=1e-12
    nor=1./sqrt
    nor=nor.reshape(1,-1)
    x_train_des=x_train_des*nor
    #########################################################
    x_train_des=np.transpose(x_train_des,(1,0))
    x_train_des=x_train_des.reshape(img_num,des_num,130)

    return x_train_des
kp_list=np.array(kp_list).astype("float32")
test_kp_list=np.array(test_kp_list).astype("float32")
des_list=normalize_extension(des_list,kp_list)
test_de_list=normalize_extension(test_de_list,test_kp_list)

####################################################################
####################### geometric extension  #######################
####################################################################

####################################################################
########################## build codebook ##########################
####################################################################
def random_init(dataset, num_centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    used = torch.zeros(num_points, dtype=torch.long)
    indices = torch.zeros(num_centers, dtype=torch.long)
    for i in range(num_centers):
        while True:
            cur_id = random.randint(0, num_points - 1)
            if used[cur_id] > 0:
                continue
            used[cur_id] = 1
            indices[i] = cur_id
            break
    indices = indices.to(device_gpu)
    centers = torch.gather(dataset.to(device_gpu), 0, indices.view(-1, 1).expand(-1, dimension))
    return centers

# Compute for each data point the closest center
def compute_codes(dataset, centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    num_centers = centers.size(0)
    # 5e8 should vary depending on the free memory on the GPU
    # Ideally, automatically ;)
    chunk_size = int(5e8 / num_centers)
    codes = torch.zeros(num_points, dtype=torch.long, device=device_gpu)
    centers_t = torch.transpose(centers, 0, 1)
    centers_norms = torch.sum(centers ** 2, dim=1).view(1, -1)
    for i in range(0, num_points, chunk_size):
        begin = i
        end = min(begin + chunk_size, num_points)
        dataset_piece = dataset[begin:end, :]
        dataset_norms = torch.sum(dataset_piece ** 2, dim=1).view(-1, 1)
        
        distances = torch.mm(dataset_piece.to(device_gpu), centers_t)
        distances *= -2.0
        distances += dataset_norms.to(device_gpu)
        distances += centers_norms
        _, min_ind = torch.min(distances, dim=1)
        codes[begin:end] = min_ind
    return codes

# Compute new centers as means of the data points forming the clusters
def update_centers(dataset, codes, num_centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    centers = torch.zeros(num_centers, dimension, dtype=torch.float, device=device_gpu)
    cnt = torch.zeros(num_centers, dtype=torch.float, device=device_gpu)   
    centers.scatter_add_(0, codes.view(-1, 1).expand(-1, dimension), dataset.to(device_gpu))
    cnt.scatter_add_(0, codes, torch.ones(num_points, dtype=torch.float, device=device_gpu))
    # Avoiding division by zero
    # Not necessary if there are no duplicates among the data points
    cnt = torch.where(cnt > 0.5, cnt, torch.ones(num_centers, dtype=torch.float, device=device_gpu))
    centers /= cnt.view(-1, 1)
    return centers

def KMeans_GPU(dataset, num_centers):
    centers = random_init(dataset, num_centers)
    codes = compute_codes(dataset, centers)
    num_iterations = 0
    while True:
        sys.stdout.write('.')
        sys.stdout.flush()
        num_iterations += 1
        centers = update_centers(dataset, codes, num_centers)
        new_codes = compute_codes(dataset, centers)
        # Waiting until the clustering stops updating altogether
        # This is too strict in practice
        #import pdb;pdb.set_trace()
        if torch.equal(codes, new_codes):
            sys.stdout.write('\n')
            print('Converged in %d iterations' % num_iterations)
            break
        codes = new_codes
    return centers, codes

def build_codebook(X, voc_size):
    #import pdb;pdb.set_trace()
    features=np.vstack((descriptor for descriptor in X))
    features=torch.from_numpy(features)
    codebook, _ = KMeans_GPU(features, voc_size)
    codebook = codebook.cpu().numpy() 
    return codebook

if os.path.exists(codebook_path):
    with open(codebook_path, 'rb') as codebook_f:
        codebook=pickle.load(codebook_f)
else:
    codebook=build_codebook(des_list, CODEBOOKSIZE) # (2050, 1024, 128)
    with open(codebook_path, 'wb') as cf:
        pickle.dump(codebook, cf)
    print('saved codeword_kmeans_geometric_1024')

####################################################################
########################## build codebook ##########################
####################################################################

import scipy.cluster.vq as vq
def improvedVLAD(X,visualDictionary,kp_list):

    predictedLabels,_ = vq.vq(X,visualDictionary)
    centers = visualDictionary
    k=1024
    #import pdb;pdb.set_trace()
    m,d = X.shape
    #V=np.zeros([k,d+1])
    V=np.zeros([k,d+1])
    #computing the differences

    # for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels==i)>0:
            expend_list=np.array([(i-200/2)/200.0])
            #expend_list=np.append(expend_list, (np.asarray(kp_list[i])-256/2)/256.0)
            # add the diferences            
            V[i]=np.append(expend_list, np.sum(X[predictedLabels==i,:]-centers[i],axis=0))
            #V[i]=np.sum(X[predictedLabels==i,:]-centers[i],axis=0)
            #import pdb;pdb.set_trace()
    V = V.flatten()                  # (1024, 128)
    # power normalization, also called square-rooting normalization
    V = np.sign(V)*np.sqrt(np.abs(V))

    # L2 normalization
    
    V = V/np.sqrt(np.dot(V,V))
    
    return V

################# des_list and test_de_list pyramid pooling #################
#des_list : (3060, 1024, 128)
#test_de_list : (1712, 1024, 128)
#for des in des_list:
#    #des=np.asarray(des)
#    des = np.resize(des, (32, 32, 128))
#    des = cutted(des, 2)
#    des = np.asarray(des)
########## VLAD는 히스토그램으로 위치를 잃지 않으니 차리리 geometry 정보를 넣어주는게 나은듯
# ----------------------------------- 중단 ----------------------------------------
################# des_list and test_de_list pyramid pooling #################
#############################################################################


train_vlad=[improvedVLAD(des_list[i], codebook, kp_list[i]) for i in range(len(des_list))] # (3060, 131072=1024*128) # des_list : (3060, 1024, 128)
test_vlad=[improvedVLAD(test_de_list[i], codebook, test_kp_list[i]) for i in range(len(test_de_list))]                        # test_de_list : (1712, 1024, 128)
#import pdb;pdb.set_trace()
#for i in range(len(train_vlad)):
    #train_vlad[i]
    
print(np.array(train_vlad).shape)
print(np.array(test_vlad).shape)

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
def svm_classifier(x_train, y_train, x_test=None, y_test=None,level=1):######## LinearSVM
    best=0
    C_range = 10.0 ** np.arange(-3, 3)# np.arange(1, 1000,1)
    #gamma_range = 10.0 ** np.arange(-3, 3)
    param_grid = dict(C=C_range.tolist())

    # Grid search for C, gamma, 5-fold CV
    print("Tuning hyper-parameters\n")
    clf = GridSearchCV(LinearSVC(random_state=0), param_grid, cv=5, n_jobs=-2)
    clf.fit(x_train, y_train)
    print(clf.best_params_)
    predict=clf.predict(x_test)
    return predict

train_vlad=np.array(train_vlad)
test_vlad=np.array(test_vlad)

test=[]
for test_path in test_paths:
    test_path=test_path.split('/')[-1]
    test.append(test_path)

#predict=svm_classifier(train_vlad, trainLabels, test_vlad)

model = LinearSVC(random_state=0, C=5)
model.fit(train_vlad, trainLabels)
predict = model.predict(test_vlad)

df = pd.DataFrame({"Id":test,"Category":predict})
df.index = np.arange(1,len(df)+1)
df.to_csv('ver3_4096_extend.csv',index=False, header=True)
