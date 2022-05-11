from sklearn.metrics import f1_score
import numpy as np
import pandas as pd


def f1_cues(y_true, y_pred):
    '''Needs flattened cues'''
    tp = sum([1 for i,j in zip(y_true, y_pred) if (i==j and i!=3)])
    fp = sum([1 for i,j in zip(y_true, y_pred) if (j!=3 and i==3)])
    fn = sum([1 for i,j in zip(y_true, y_pred) if (i!=3 and j==3)])
    if tp==0:
        prec = 0.0001
        rec = 0.0001
    else:
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {2*prec*rec/(prec+rec)}")
    return prec, rec, 2*prec*rec/(prec+rec)
    
    
def f1_scope(y_true, y_pred, level = 'token'): #This is for gold cue annotation scope, thus the precision is always 1.
    if level == 'token':
        print(f1_score([i for i in j for j in y_true], [i for i in j for j in y_pred]))
    elif level == 'scope':
        tp = 0
        fn = 0
        fp = 0
        for y_t, y_p in zip(y_true, y_pred):
            if y_t == y_p:
                tp+=1
            else:
                fn+=1
        prec = 1
        rec = tp/(tp+fn)
        print(f"Precision: {prec}")
        print(f"Recall: {rec}")
        print(f"F1 Score: {2*prec*rec/(prec+rec)}")

def report_per_class_accuracy(y_true, y_pred):
    labels = list(np.unique(y_true))
    lab = list(np.unique(y_pred))
    labels = list(np.unique(labels+lab))
    n_labels = len(labels)
    data = pd.DataFrame(columns = labels, index = labels, data = np.zeros((n_labels, n_labels)))
    for i,j in zip(y_true, y_pred):
        data.at[i,j]+=1
    print(data)
    
def flat_accuracy(preds, labels, input_mask = None):
    pred_flat = [i for j in preds for i in j]
    labels_flat = [i for j in labels for i in j]
    # print(pred_flat)
    # print(labels_flat)
    return sum([1 if i==j else 0 for i,j in zip(pred_flat,labels_flat)]) / len(labels_flat)
    

def flat_accuracy_positive_cues(preds, labels, input_mask = None):
    pred_flat = [i for i,j in zip([i for j in preds for i in j],[i for j in labels for i in j]) if (j!=4 and j!=3)]
    labels_flat = [i for i in [i for j in labels for i in j] if (i!=4 and i!=3)]
    if len(labels_flat) != 0:
        return sum([1 if i==j else 0 for i,j in zip(pred_flat,labels_flat)]) / len(labels_flat)
    else:
        return None

def scope_accuracy(preds, labels):
    correct_count = 0
    count = 0
    for i,j in zip(preds, labels):
        if i==j:
            correct_count+=1
        count+=1
    return correct_count/count
