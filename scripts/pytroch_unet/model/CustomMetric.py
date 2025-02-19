import torch
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score

eps=1e-10

def recall_m(kwargs):
    y_pred = kwargs['y_pred']
    y_true = kwargs['y_true']
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + eps)
    return recall

 # Calculate precision
def precision_m(kwargs):
    y_pred = kwargs['y_pred']
    y_true = kwargs['y_true']
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + eps)
    return precision

 # Calculate F1 score based on precision and recall
def f1_m(kwargs):
    y_pred = kwargs['y_pred']
    y_true = kwargs['y_true']
    precision = precision_m({'y_true':y_true, 'y_pred':y_pred})
    recall = recall_m({'y_true':y_true, 'y_pred':y_pred})
    return (2 * ((precision * recall) / (precision + recall + eps)))

def sk_recall(kwargs):
    y_pred = kwargs['y_pred']
    y_true = kwargs['y_true']
    threshold = kwargs['threshold']
    y_pred = np.where(y_pred>threshold,1,0)
    return recall_score(y_true,y_pred, zero_division=0)

def sk_precision(kwargs):
    y_pred = kwargs['y_pred']
    y_true = kwargs['y_true']
    threshold = kwargs['threshold']
    y_pred = np.where(y_pred>threshold,1,0)
    return precision_score(y_true,y_pred, zero_division=0)

def sk_f1score(kwargs):
    y_pred = kwargs['y_pred']
    y_true = kwargs['y_true']
    threshold = kwargs['threshold']
    y_pred = np.where(y_pred>threshold,1,0)
    return f1_score(y_true,y_pred, zero_division=0)
