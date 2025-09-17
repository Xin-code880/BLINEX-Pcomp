import torch
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

def prediction(args, alpha, alpha1, gamma, xp, xn, xt, yt):
    Result = []
    TP, TN, FP, FN = 0, 0, 0, 0

    yt_test = yt
    for i, (xp_test, xn_test, xt_test) in enumerate(zip(xp, xn,xt)):
        K_test = pairwise_kernels(xp_test, xt_test, metric='linear')
        K1_test = pairwise_kernels(xn_test, xt_test, metric='linear')
        result = alpha[i].T @ K_test + alpha1[i].T @ K1_test
        Result.append(result)
    results = sum([Result[i] * gamma[i] for i in range(args.m)])
    result_data_numpy = np.array(results.data)
    result_data_tensor = torch.tensor(result_data_numpy)
    predicted = (result_data_tensor >= 0).float()
    predicted[predicted == 0] = -1.0
    predicted = predicted.cpu().numpy()
    yt_test = yt_test.cpu().numpy() if isinstance(yt_test, torch.Tensor) else yt_test

    TP += np.sum((predicted == 1) & (yt_test == 1)).sum().item()
    TN += np.sum((predicted == -1) & (yt_test == -1)).sum().item()
    FP += np.sum((predicted == 1) & (yt_test == -1)).sum().item()
    FN += np.sum((predicted == -1) & (yt_test == 1)).sum().item()

    acc = (TP + TN) / results.size if results.size > 0 else 0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    F1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    Gmean = np.sqrt(recall * specificity) if (recall > 0 and specificity > 0) else 0

    return acc, prec, recall, specificity, F1, Gmean
