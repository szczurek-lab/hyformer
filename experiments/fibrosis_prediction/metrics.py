import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


def get_test_metrics(predictions, targets, threshold=0.5):
    test_metrics = {}
    test_metrics['auroc'] = rocauc(predictions, targets)
    test_metrics['auprc'] = prauc(predictions, targets)
    test_metrics['precision'] = precision(predictions, targets, k=None, threshold=threshold)
    test_metrics['precision_at_100'] = precision(predictions, targets, k=100, threshold=threshold)
    test_metrics['bedroc_20'] = bedroc(predictions, targets, alpha=20)
    test_metrics['recall'] = recall(predictions, targets, k=None, threshold=threshold)
    test_metrics['recall_at_100'] = recall(predictions, targets, k=100, threshold=threshold)
    test_metrics['accuracy'] = accuracy(predictions, targets, threshold=threshold)
    return test_metrics


def accuracy(y_pred, y_true, threshold, *args, **kwargs):
    return accuracy_score(y_true=y_true, y_pred=y_pred >= threshold, *args, **kwargs).item()


def bedroc(y_pred, y_true, alpha):
    """BEDROC metric implemented according to Truchon and Bayley.

    The Boltzmann Enhanced Descrimination of the Receiver Operator
    Characteristic (BEDROC) score is a modification of the Receiver Operator
    Characteristic (ROC) score that allows for a factor of *early recognition*.

    References:
        The original paper by Truchon et al. is located at `10.1021/ci600426e
        <http://dx.doi.org/10.1021/ci600426e>`_.
    Source: 
        https://scikit-chem.readthedocs.io/en/latest/_modules/skchem/metrics.html#:~:text=The%20Boltzmann%20Enhanced%20Descrimination%20of%20the%20Receiver%20Operator,that%20allows%20for%20a%20factor%20of%20%2Aearly%20recognition%2A.

    Args:
        y_true (array_like):
            Binary class labels. 1 for positive class, 0 otherwise.
        y_pred (array_like):
            Prediction values.
        alpha (float):
            Early recognition parameter.

    Returns:
        float:
            Value in interval [0, 1] indicating degree to which the predictive
            technique employed detects (early) the positive class.
    """

    big_n = len(y_true)
    n = sum(y_true == 1)
    order = np.argsort(-y_pred)
    m_rank = (y_true[order] == 1).nonzero()[0]
    s = np.sum(np.exp(-alpha * m_rank / big_n))
    r_a = n / big_n
    rand_sum = r_a * (1 - np.exp(-alpha))/(np.exp(alpha/big_n) - 1)
    fac = r_a * np.sinh(alpha / 2) / (np.cosh(alpha / 2) - np.cosh(alpha/2 - alpha * r_a))
    cte = 1 / (1 - np.exp(alpha * (1 - r_a)))
    return float(s * fac / rand_sum + cte)


def prauc(y_pred, y_true, *args, **kwargs):
    return average_precision_score(y_score=y_pred, y_true=y_true, *args, **kwargs).item()


def precision(y_pred, y_true, k, threshold, *args, **kwargs):
    if k is not None:
        assert len(y_pred) >= k
        idx_top_k = np.argsort(y_pred, axis=0)[-k:][::-1].flatten()
        return precision_score(y_pred=y_pred[idx_top_k]>=threshold, y_true=y_true[idx_top_k]).item()
    else:
        return precision_score(y_pred=y_pred>=threshold, y_true=y_true).item()
    

def recall(y_pred, y_true, k, threshold, *args, **kwargs):
    if k is not None:
        assert len(y_pred) >= k
        idx_top_k = np.argsort(y_pred, axis=0)[-k:][::-1].flatten()
        return recall_score(y_pred=y_pred[idx_top_k]>=threshold, y_true=y_true[idx_top_k]).item()
    else:
        return recall_score(y_pred=y_pred>=threshold, y_true=y_true).item()
    

def rocauc(y_pred, y_true, *args, **kwargs):
    return roc_auc_score(y_score=y_pred, y_true=y_true, *args, **kwargs).item()
