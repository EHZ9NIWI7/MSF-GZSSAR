# Thanks to P Gupta for the released code on github (https://github.com/skelemoa/synse-zsl)
import argparse
import pickle as pkl

import numpy as np
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser(description='View adaptive')
parser.add_argument('--ss', type=int, help="split size")
parser.add_argument('--st', type=str, help="split type")
parser.add_argument('--dataset', type=str, help="dataset path")
parser.add_argument('--wdir', type=str, help="directory to save weights path")
parser.add_argument('--le', type=str, help="language embedding model")
parser.add_argument('--ve', type=str, help="visual embedding model")
parser.add_argument('--phase', type=str, help="train or val")
parser.add_argument('--ntu', type=int, help="number of classes")
parser.add_argument('--tm', type=str, help='text mode')
parser.add_argument('--th', type=int, default=0, help='threshold')
parser.add_argument('--t', type=int, default=0, help='temp')
args = parser.parse_args()

ss = args.ss
st = args.st
dataset = args.dataset
wdir = args.wdir
le = args.le
ve = args.ve
phase = args.phase
num_classes = args.ntu
tm = args.tm
th_set = args.th
t_set = args.t

seed = 5
np.random.seed(seed)

    
def temp_scale(seen_features, T): # softmax
    return np.array([np.exp(i)/np.sum(np.exp(i)) for i in (seen_features + 1e-12)/T])

unseen_zs = np.load(f'{wdir}/{le}/{tm}/MSF_{str(ss)}_r_unseen_zs.npy')    # (2000, 5)     (1440, 12)
seen_zs = np.load(f'{wdir}/{le}/{tm}/MSF_{str(ss)}_r_seen_zs.npy')        # (2000, 5)     (1440, 12)
unseen_train = np.load(f'{dataset}/ztest_out.npy')                          # (1367, 60)    (3284, 60)
seen_train = np.load(f'{dataset}/val_out.npy')                              # (2000, 60)    (1440, 60)

for f in [ss]:
    best_model = None
    best_acc = 0
    best_thresh = 0
    t_iter = [i for i in range(1,10)] if t_set == 0 else [t_set] # range(1, 10) -> (1, 4)
    for t in t_iter:
        fin_val_acc = 0
        fin_train_acc = 0
        prob_unseen_zs = unseen_zs                                          # (2000, 5)     (1440, 12)
        prob_unseen_train = temp_scale(unseen_train, t)                     # (1367, 60)    (3284, 60)
        prob_seen_zs = seen_zs                                              # (2000, 5)     (1440, 12)
        prob_seen_train = temp_scale(seen_train, t)                         # (2000, 60)    (1440, 60)

        feat_unseen_zs = np.sort(prob_unseen_zs, 1)[:,::-1][:,:f]           # (2000, 5)     (1440, 12)  从大到小
        feat_unseen_train = np.sort(prob_unseen_train, 1)[:,::-1][:,:f]     # (1367, 5)     (3284, 60)
        feat_seen_zs = np.sort(prob_seen_zs, 1)[:,::-1][:,:f]               # (2000, 5)     (1440, 12)
        feat_seen_train = np.sort(prob_seen_train, 1)[:,::-1][:,:f]         # (2000, 5)     (1440, 12)
        # feat_unseen_train.shape[0] // 2   ,   feat_seen_train.shape[0] // 2
        val_unseen_inds = np.random.choice(np.arange(feat_unseen_train.shape[0]), feat_unseen_train.shape[0] // 5, replace=False)   # choose 300 from 3284
        val_seen_inds = np.random.choice(np.arange(feat_seen_train.shape[0]), feat_seen_train.shape[0] // 5, replace=False)       # choose 400 from 1440
        train_unseen_inds = np.array(list(set(list(np.arange(feat_unseen_train.shape[0]))) - set(list(val_unseen_inds)))) # 1984
        train_seen_inds = np.array(list(set(list(np.arange(feat_seen_train.shape[0]))) - set(list(val_seen_inds))))       # 1040

        gating_train_x = np.concatenate([np.concatenate([feat_unseen_zs[train_unseen_inds, :], feat_unseen_train[train_unseen_inds, :]], 1), np.concatenate([feat_seen_zs[train_seen_inds, :], feat_seen_train[train_seen_inds, :]], 1)], 0)
        gating_train_y = [0]*len(train_unseen_inds) + [1]*len(train_seen_inds)
        gating_val_x = np.concatenate([np.concatenate([feat_unseen_zs[val_unseen_inds, :], feat_unseen_train[val_unseen_inds, :]], 1), np.concatenate([feat_seen_zs[val_seen_inds, :], feat_seen_train[val_seen_inds, :]], 1)], 0)
        gating_val_y = [0]*len(val_unseen_inds) + [1]*len(val_seen_inds)

        train_inds = np.arange(gating_train_x.shape[0])
        np.random.shuffle(train_inds)

        model = LogisticRegression(random_state=0, C=1, solver='lbfgs', n_jobs=-1,
                                     multi_class='multinomial', verbose=0, max_iter=5000,
                                     ).fit(gating_train_x[train_inds, :], np.array(gating_train_y)[train_inds])
        prob = model.predict_proba(gating_val_x)
        best = 0
        bestT = 0
        th_iter = [i for i in range(25, 75, 1)] if th_set == 0 else [th_set] # (25, 75) -> (45, 55)
        for th in th_iter:
            y = prob[:, 0] > th/100
            acc = np.sum((1 - y) == gating_val_y)/len(gating_val_y)
            if acc > best:
                best = acc
                bestT = th/100
        fin_val_acc += best
        pred_train = model.predict(gating_train_x)
        train_acc = np.sum(pred_train == gating_train_y)/len(gating_train_y)
        fin_train_acc += train_acc
        
        if fin_val_acc > best_acc:
            best_temp = t
            best_acc = fin_val_acc
            best_thresh = bestT
            best_model = model
    print('best validation accuracy for the gating model', best_acc)
    print('best threshold', best_thresh)
    print('best temperature', best_temp)

with open(wdir.replace('_val', '') + f'/{le}/{tm}/gating_model.pkl', 'wb') as f:
    pkl.dump(best_model, f)
    f.close()
