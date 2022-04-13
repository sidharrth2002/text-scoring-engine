import numpy as np
from scipy.special import softmax
import shutil

from transformers import EvalPrediction

from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
)

def calc_classification_metrics(p: EvalPrediction):
    pred_labels = np.argmax(p.predictions[0], axis=1)
    print(pred_labels)
    pred_scores = softmax(p.predictions[0], axis=1)
    print(pred_scores)
    # labels = p.label_ids
    # acc = (pred_labels == labels).mean()
    # f1 = f1_score(y_true=labels, y_pred=pred_labels, average='weighted')
    result = {
      "acc": 0,
      "f1": 0,
      "acc_and_f1": 0,
      "mcc": 0,
      "QWK": 0
    }
    # print(result)
    return result

def get_score(p: EvalPrediction):
  pred_label = np.argmax(p.predictions[0], axis=1)[0]
  return {
    "score": pred_label
  }
