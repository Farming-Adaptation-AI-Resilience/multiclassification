import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import math
from os.path import isfile, join
from os import listdir
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, jaccard_score

#Good guide
#https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
def to_percent(value):
    return value*100

def space():
    print('\n\n')

def metrics(tn, fp, fn, tp):
    P = tp + fn
    N = fp + tn
    not_P = tp + fp
    not_N = fn + tn
    print('Total amount of pixels:')
    total = P+N
    print(P+N)
    print('For true values:')
    print('Forests = {}'.format(P))
    print('Forests = {}%'.format(to_percent(P/total)))
    print('Not-Forests = {}'.format(N))
    print('Not-Forests = {}%'.format(to_percent(N/total)))
    print('\nFor predicted values:')
    print('Forests = {}'.format(not_P))
    print('Forests = {}%'.format(to_percent(not_P/total)))
    print('Not-Forests = {}'.format(not_N))
    print('Not-Forests = {}%'.format(to_percent(not_N/total)))
    space()
    print('TP: {}'.format(tp))
    print('FP: {}'.format(fp))
    print('TN: {}'.format(tn))
    print('FN: {}'.format(fn))
    space()
    accuracy_cal = (tp+tn)/ (P+N)
    print('Accuracy:')
    print(to_percent(accuracy_cal))
    recall = tp/P
    specificity = tn/N
    precision = tp/(tp+fp)
    f_score = (2*precision*recall)/(precision+recall)
    jaccard = tp/(tp+fn+fp)
    #auc = metrics.auc(recall, specificity)
    #mcc = ((tp*tn) - (fp*fn)) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    print('recall:')
    print(to_percent(recall))
    print('specificity:')
    print(to_percent(specificity))
    print('precision:')
    print(to_percent(precision))
    print('f_score:')
    print(to_percent(f_score))
    print('jaccard:')
    print(to_percent(jaccard))
#    print('AUC (area under curve)')
#    print(to_percent(auc))
    #print('MCC: ') 
    #print(mcc) 


def read_files(files,pred_dir,gt_folder):
    #for i in files['name']:
    for i in files:
        if(i%100 == 0):
            print(i)
        
        y_true = cv2.imread(gt_folder + 'ground_truth_training_' + str(i) + '.png')
        y_true = y_true[:,:,0]
        #y_pred = plt.imread('Predictions/Test_2_Rahel/training_' + str(i) + '.TIF')
        #y_pred = plt.imread('../../Final_cnn/5_fold/predictions/training_' + str(i) + '.TIF')
        y_pred = cv2.imread(pred_dir+ str(i) + '.png')
        y_pred = y_pred[:,:,0]
        y_pred = np.where(y_pred==85,60,y_pred)
        y_pred = np.where(y_pred==170,255,y_pred)
        '''
        print('y_true')
        print(np.unique(y_true))
        print(y_true.shape)
        print('y_pred')
        print(np.unique(y_pred))
        print(y_pred.shape)
        '''
        print(np.unique(y_true==y_pred))
        #if(i>=727):
        #    print(i)
        #    print(y_pred[0,:])
        #    print(y_true[0,:])
        y_true = y_true.flatten().tolist()
        y_pred = y_pred.flatten().tolist()
        
        #tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,60,255]).ravel()
        print(confusion_matrix(y_true, y_pred, labels=[0,60,255]).ravel())
        print('accuracy:')
        print(accuracy_score(y_true, y_pred))
        print('f1_score micro:')
        print(f1_score(y_true, y_pred,average='micro'))
        print('f1_score macro:')
        print(f1_score(y_true, y_pred,average='macro'))
        print('f1_score weighted:')
        print(f1_score(y_true, y_pred,average='weighted'))
        print('jaccard index micro:')
        print(jaccard_score(y_true, y_pred,average='micro'))
        print('jaccard index macro:')
        print(jaccard_score(y_true, y_pred,average='macro'))
        print('jaccard index weighted:')
        print(jaccard_score(y_true, y_pred,average='weighted'))

        #print(tn,fp,fn,tp)
        '''
        TN += tn
        FP += fp
        FN += fn
        TP += tp
        '''
        break
    #accuracy = accuracy_score(y_true,y_pred)
    #print(accuracy*100)
    '''
    print(TN)
    print(FP)
    print(FN)
    print(TP)
    metrics(TN, FP, FN, TP)
    '''

TN = FP = FN = TP = 0
epochs = 150
person = 'Keanu'
print('Number of files the training set was trained on {}'.format(2400))
print('Number of epochs for training is {}'.format(epochs))
predictions = '300_epochs_predictions/4_epochs_predictions/' 
gt_folder = 'test_partition_ground_truth_visible/'
files = [int(f.split('.')[0]) for f in listdir(predictions) if isfile(join(predictions, f))]
files.sort()
print(files)
read_files(files,predictions,gt_folder)
'''
for i in files['name']:
    i = int(i.split('_')[1])
    if(i%100 == 0):
        print(i)
    y_true = plt.imread('train_gt/gt_training_' + str(i) + '.TIF')
    #y_pred = plt.imread('Predictions/Test_2_Rahel/training_' + str(i) + '.TIF')
    y_pred = plt.imread('../../Final_cnn/5_fold/predictions/training_' + str(i) + '.TIF')
    y_pred = y_pred[:,:,0]
    #if(i>=727):
    #    print(i)
    #    print(y_pred[0,:])
    #    print(y_true[0,:])
    y_true = y_true.flatten().tolist()
    y_pred = y_pred.flatten().tolist()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,255]).ravel()
    TN += tn
    FP += fp
    FN += fn
    TP += tp
#    break
#accuracy = accuracy_score(y_true,y_pred)
#print(accuracy*100)
print(TN)
print(FP)
print(FN)
print(TP)
metrics(TN, FP, FN, TP)
'''
