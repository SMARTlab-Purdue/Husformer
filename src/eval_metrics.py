import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score


def multiclass_acc(preds, truths):
    multi_acc = np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

    return multi_acc


def mae1(results, truths, exclude_zero=False):
    test_preds1 = results.view(-1).cpu().detach().numpy()
    for i,j in enumerate(test_preds1):
        if -1 < j < 0:
            test_preds1[i] = -1
        if 0 < j < 1:
            test_preds1[i] = 1
    test_preds1 = np.clip(test_preds1, a_min=-1., a_max=2.)
    test_preds = np.around(test_preds1)
    test_truth = truths.view(-1).cpu().detach().numpy()
    mae = np.mean(np.absolute(test_preds - test_truth))
    return mae

def eval_hus(results, truths, exclude_zero=False):

    test_preds1 = results.view(-1).cpu().detach().numpy()
    for i,j in enumerate(test_preds1):
        if -1 < j < 0:
            test_preds1[i] = -1
        if 0 < j < 1:
            test_preds1[i] = 1
    test_preds1 = np.clip(test_preds1, a_min=-1., a_max=2.)
    test_preds = np.around(test_preds1)
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)

    _, _, f1, _ = precision_recall_fscore_support(test_preds[non_zeros], test_truth[non_zeros], average='weighted')
    print("-" * 50)
    print("MAE: ", mae)
    print("Correlation Coefficient: ", corr)
    print("mult_acc: ", mult_a5)
    print('f1_score:', f1)
    print("-" * 50)




