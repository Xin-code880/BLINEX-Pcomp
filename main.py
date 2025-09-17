import argparse
from data_composed import  generate_pcomp_data
from model import training
import numpy as np
from predict import prediction

objective_values = []
parser = argparse.ArgumentParser()
parser.add_argument('-ds', help='specify a dataset', default='usps', type=str, required=False)
parser.add_argument('-uci', help = 'Is UCI datasets?', default=1, type=int, choices=[0,1], required=False)
parser.add_argument('-prior', help = 'class (positive) prior', default=0.5, type=float, required=False)
parser.add_argument('-gpu', help = 'used gpu id', default='0', type=str, required=False)
parser.add_argument('-m', help = 'views number', default=1, type=int, required=False)
parser.add_argument('-n', help = 'number of unlabeled data pairs', default=1000, type=int, required=False)
parser.add_argument('-g', help = 'rbf', default=1, type=float, required=False)
parser.add_argument('-kernel', help = 'linear or rbf', default='linear', type=str, required=False)
args = parser.parse_args()

args.m = 1
args.ds = 'mnist'
args.prior = 0.5

pai_plus = args.prior
pai_minus = 1 - args.prior
a = 0.1
b = 5
C = 10
C2 = 10
lamda = 2
mu = 0.3
eps = 0.2

if args.ds == 'mnist' or args.ds =='kmnist' or args.ds == 'fashion' or args.ds == 'cifar':
    args.uci = 0
    args.n = 2000
    args.kernel = 'linear'
elif args.ds == 'bank' or 'cnae9':
    args.uci = 1
    args.kernel = 'rbf'
    args.n = 200
    args.g = 1 / args.n


xp, xn, real_yp, real_yn, given_yp, given_yn, xt, yt, dim = generate_pcomp_data(args)
features_1 = xp
features_2 = xn
features_test = xt

alpha, alpha1,gamma = training(args, features_1, features_2, C, a, b, C2, args.m, lamda, mu, pai_plus, pai_minus, eps)
acc, prec, recall, specificity, F1, gmean = prediction(args, alpha, alpha1, gamma, xp, xn, xt, yt)
print(f'ds:{args.ds}, prior:{args.prior}')
print('Acc:',acc)
print('Prec:',prec)
print('Recall:',recall)
print('Specificity:',specificity)
print('F1-score:',F1)
print('Gmean:',gmean)
