from typing import List


def recall_at_k(labels: List[int], scores: List[float], k=5) -> float:
    tp = [x[0] for x in sorted(list(zip(list(map(float, labels)), scores)),  
                               key=lambda x: x[1])[::-1][:k] if x[0] == 1].count(1)
    fn = [x[0] for x in sorted(list(zip(list(map(float, labels)), scores)),
                               key=lambda x: x[1])[::-1][k:] if x[0] == 1].count(1)

    recall_k = tp / (tp + fn)

    return recall_k


def precision_at_k(labels: List[int], scores: List[float], k=5) -> float:
    tp = [x[0] for x in sorted(list(zip(list(map(float, labels)), scores)),  
                               key=lambda x: x[1])[::-1][:k] if x[0] == 1].count(1)
    fp = [x[0] for x in sorted(list(zip(list(map(float, labels)), scores)),  
                               key=lambda x: x[1])[::-1][:k] if x[0] == 0].count(0)

    precision_k = tp / (tp + fp)

    return precision_k


def specificity_at_k(labels: List[int], scores: List[float], k=5) -> float:
    fp = [x[0] for x in sorted(list(zip(list(map(float, labels)), scores)),  
                               key=lambda x: x[1])[::-1][:k] if x[0] == 0].count(0)
    tn = [x[0] for x in sorted(list(zip(list(map(float, labels)), scores)),  
                               key=lambda x: x[1])[::-1][k:] if x[0] == 0].count(0)
    if fp + tn == 0:
        specificity_k = tn / (tn + fp + 1e-16)
    else:
        specificity_k = tn / (tn + fp)

    return specificity_k

def f1_at_k(labels: List[int], scores: List[float], k=5) -> float:
    tp = [x[0] for x in sorted(list(zip(list(map(float, labels)), scores)),  
                               key=lambda x: x[1])[::-1][:k] if x[0] == 1].count(1)
    fn = [x[0] for x in sorted(list(zip(list(map(float, labels)), scores)),
                               key=lambda x: x[1])[::-1][k:] if x[0] == 1].count(1)
    fp = [x[0] for x in sorted(list(zip(list(map(float, labels)), scores)),  
                               key=lambda x: x[1])[::-1][:k] if x[0] == 0].count(0)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if precision + recall == 0:
        f1_k = 2 * precision * recall / (precision + recall + 1e-16)
    else:
        f1_k = 2 * precision * recall / (precision + recall)


    return f1_k