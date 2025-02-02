"""Evaluate DFR on spurious correlations datasets."""

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import tqdm
import argparse
import sys
from collections import defaultdict
import json
from functools import partial
import pickle
import random
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from wb_data import WaterBirdsDataset, get_loader, get_transform_cub, log_data
from utils import Logger, AverageMeter, set_seed, evaluate, get_y_p
from metashift_data import get_metashift_loaders, prepare_confounder_data
from domino_data import DominoeMnistCifarDataset

# WaterBirds
C_OPTIONS = [1., 0.7, 0.3, 0.1, 0.07, 0.03, 0.01]
CLASS_WEIGHT_OPTIONS = [1., 2., 3., 10., 100., 300., 1000.]
# CelebA
REG = "l1"
# # REG = "l2"
# C_OPTIONS = [3., 1., 0.3, 0.1, 0.03, 0.01, 0.003]
# CLASS_WEIGHT_OPTIONS = [1., 2., 3., 10., 100, 300, 500]

CLASS_WEIGHT_OPTIONS = [{0: 1, 1: w} for w in CLASS_WEIGHT_OPTIONS] + [
        {0: w, 1: 1} for w in CLASS_WEIGHT_OPTIONS]

# set random seed
SEED = 50
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed (seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
set_seed(SEED)



parser = argparse.ArgumentParser(description="Tune and evaluate DFR.")
parser.add_argument(
    "--data_dir", type=str,
    default='/home/f_hosseini/parsa/data/Domino',
    help="Train dataset directory")
parser.add_argument(
    "--result_path", type=str, default="logs/log1.pkl",
    help="Path to save results")
parser.add_argument(
    "--ckpt_path", type=str, default='/home/f_hosseini/parsa/models/Dominos_ResNet50_spuriousity=0.90.pt',
    help="Checkpoint path")
parser.add_argument(
    "--batch_size", type=int, default=100, required=False,
    help="Batch Size")
parser.add_argument(
    "--balance_dfr_val", type=bool, default=True, required=False,
    help="Subset validation to have equal groups for DFR(Val)")
parser.add_argument(
    "--notrain_dfr_val", type=bool, default=True, required=False,
    help="Do not add train data for DFR(Val)")
parser.add_argument(
    "--tune_class_weights_dfr_train", action='store_true',
    help="Learn class weights for DFR(Train)")
parser.add_argument(
    "--dataset", type=str, default='waterbirds',
    help="Experiment Dataset")
args = parser.parse_args()


def dfr_on_validation_tune(
        all_embeddings, all_y, all_g, preprocess=True,
        balance_val=False, add_train=True, num_retrains=1):

    worst_accs = {}
    for i in range(num_retrains):
        x_val = all_embeddings["val"]
        y_val = all_y["val"]
        g_val = all_g["val"]
        n_groups = np.max(g_val) + 1

        n_val = len(x_val) // 2
        idx = np.arange(len(x_val))
        np.random.shuffle(idx)

        x_valtrain = x_val[idx[n_val:]]
        y_valtrain = y_val[idx[n_val:]]
        g_valtrain = g_val[idx[n_val:]]

        n_groups = np.max(g_valtrain) + 1
        g_idx = [np.where(g_valtrain == g)[0] for g in range(n_groups)]
        min_g = np.min([len(g) for g in g_idx])
        for g in g_idx:
            np.random.shuffle(g)
        if balance_val:
            x_valtrain = np.concatenate([x_valtrain[g[:min_g]] for g in g_idx])
            y_valtrain = np.concatenate([y_valtrain[g[:min_g]] for g in g_idx])
            g_valtrain = np.concatenate([g_valtrain[g[:min_g]] for g in g_idx])

        x_val = x_val[idx[:n_val]]
        y_val = y_val[idx[:n_val]]
        g_val = g_val[idx[:n_val]]

        n_train = len(x_valtrain) if add_train else 0

        x_train = np.concatenate([all_embeddings["train"][:n_train], x_valtrain])
        y_train = np.concatenate([all_y["train"][:n_train], y_valtrain])
        g_train = np.concatenate([all_g["train"][:n_train], g_valtrain])
        print(np.bincount(g_train))
        if preprocess:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_val = scaler.transform(x_val)


        if balance_val and not add_train:
            cls_w_options = [{0: 1., 1: 1.}]
        else:
            cls_w_options = CLASS_WEIGHT_OPTIONS
        for c in C_OPTIONS:
            for class_weight in cls_w_options:
                logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",
                                            class_weight=class_weight)
                logreg.fit(x_train, y_train)
                preds_val = logreg.predict(x_val)
                group_accs = np.array(
                    [(preds_val == y_val)[g_val == g].mean()
                     for g in range(n_groups)])
                worst_acc = np.min(group_accs)
                if i == 0:
                    worst_accs[c, class_weight[0], class_weight[1]] = worst_acc
                else:
                    worst_accs[c, class_weight[0], class_weight[1]] += worst_acc
                # print(c, class_weight[0], class_weight[1], worst_acc, worst_accs[c, class_weight[0], class_weight[1]])
    ks, vs = list(worst_accs.keys()), list(worst_accs.values())
    best_hypers = ks[np.argmax(vs)]
    return best_hypers


def dfr_on_validation_eval(
        c, w1, w2, all_embeddings, all_y, all_g, num_retrains=20,
        preprocess=True, balance_val=False, add_train=True):
    coefs, intercepts = [], []
    if preprocess:
        scaler = StandardScaler()
        scaler.fit(all_embeddings["train"])

    for i in range(num_retrains):
        x_val = all_embeddings["val"]
        y_val = all_y["val"]
        g_val = all_g["val"]
        n_groups = np.max(g_val) + 1
        g_idx = [np.where(g_val == g)[0] for g in range(n_groups)]
        min_g = np.min([len(g) for g in g_idx])
        for g in g_idx:
            np.random.shuffle(g)
        if balance_val:
            x_val = np.concatenate([x_val[g[:min_g]] for g in g_idx])
            y_val = np.concatenate([y_val[g[:min_g]] for g in g_idx])
            g_val = np.concatenate([g_val[g[:min_g]] for g in g_idx])

        n_train = len(x_val) if add_train else 0
        train_idx = np.arange(len(all_embeddings["train"]))
        np.random.shuffle(train_idx)
        train_idx = train_idx[:n_train]

        x_train = np.concatenate(
            [all_embeddings["train"][train_idx], x_val])
        y_train = np.concatenate([all_y["train"][train_idx], y_val])
        g_train = np.concatenate([all_g["train"][train_idx], g_val])
        print(np.bincount(g_train))
        if preprocess:
            x_train = scaler.transform(x_train)

        logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",
                                    class_weight={0: w1, 1: w2})
        logreg.fit(x_train, y_train)
        coefs.append(logreg.coef_)
        intercepts.append(logreg.intercept_)

    x_test = all_embeddings["test"]
    y_test = all_y["test"]
    g_test = all_g["test"]
    print(np.bincount(g_test))

    if preprocess:
        x_test = scaler.transform(x_test)
    logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",
                                class_weight={0: w1, 1: w2})
    n_classes = np.max(y_train) + 1
    # the fit is only needed to set up logreg
    logreg.fit(x_train[:n_classes], np.arange(n_classes))
    logreg.coef_ = np.mean(coefs, axis=0)
    logreg.intercept_ = np.mean(intercepts, axis=0)
    preds_test = logreg.predict(x_test)
    preds_train = logreg.predict(x_train)
    n_groups = np.max(g_train) + 1
    test_accs = [(preds_test == y_test)[g_test == g].mean()
                 for g in range(n_groups)]
    test_mean_acc = (preds_test == y_test).mean()
    train_accs = [(preds_train == y_train)[g_train == g].mean()
                  for g in range(n_groups)]
    return test_accs, test_mean_acc, train_accs


def dfr_train_subset_tune(
        all_embeddings, all_y, all_g, preprocess=True,
        learn_class_weights=False):

    x_val = all_embeddings["val"]
    y_val = all_y["val"]
    g_val = all_g["val"]

    x_train = all_embeddings["train"]
    y_train = all_y["train"]
    g_train = all_g["train"]

    if preprocess:
        scaler = StandardScaler()
        scaler.fit(x_train)

    n_groups = np.max(g_train) + 1
    g_idx = [np.where(g_train == g)[0] for g in range(n_groups)]
    for g in g_idx:
        np.random.shuffle(g)
    min_g = np.min([len(g) for g in g_idx])
    x_train = np.concatenate([x_train[g[:min_g]] for g in g_idx])
    y_train = np.concatenate([y_train[g[:min_g]] for g in g_idx])
    g_train = np.concatenate([g_train[g[:min_g]] for g in g_idx])
    print(np.bincount(g_train))
    if preprocess:
        x_train = scaler.transform(x_train)
        x_val = scaler.transform(x_val)

    worst_accs = {}
    if learn_class_weights:
        cls_w_options = CLASS_WEIGHT_OPTIONS
    else:
        cls_w_options = [{0: 1., 1: 1.}]
    for c in C_OPTIONS:
        for class_weight in cls_w_options:
            logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",
                                        class_weight=class_weight, max_iter=20)
            logreg.fit(x_train, y_train)
            preds_val = logreg.predict(x_val)
            group_accs = np.array(
                [(preds_val == y_val)[g_val == g].mean() for g in range(n_groups)])
            worst_acc = np.min(group_accs)
            worst_accs[c, class_weight[0], class_weight[1]] = worst_acc
            print(c, class_weight, worst_acc, group_accs)

    ks, vs = list(worst_accs.keys()), list(worst_accs.values())
    best_hypers = ks[np.argmax(vs)]
    return best_hypers


def dfr_train_subset_eval(
        c, w1, w2, all_embeddings, all_y, all_g, num_retrains=10,
        preprocess=True):
    coefs, intercepts = [], []
    x_train = all_embeddings["train"]
    scaler = StandardScaler()
    scaler.fit(x_train)

    for i in range(num_retrains):
        x_train = all_embeddings["train"]
        y_train = all_y["train"]
        g_train = all_g["train"]
        n_groups = np.max(g_train) + 1

        g_idx = [np.where(g_train == g)[0] for g in range(n_groups)]
        min_g = np.min([len(g) for g in g_idx])
        for g in g_idx:
            np.random.shuffle(g)
        x_train = np.concatenate([x_train[g[:min_g]] for g in g_idx])
        y_train = np.concatenate([y_train[g[:min_g]] for g in g_idx])
        g_train = np.concatenate([g_train[g[:min_g]] for g in g_idx])
        print(np.bincount(g_train))

        if preprocess:
            x_train = scaler.transform(x_train)

        logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",
                                        class_weight={0: w1, 1: w2})
        logreg.fit(x_train, y_train)

        coefs.append(logreg.coef_)
        intercepts.append(logreg.intercept_)

    x_test = all_embeddings["test"]
    y_test = all_y["test"]
    g_test = all_g["test"]

    if preprocess:
        x_test = scaler.transform(x_test)

    logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear")
    n_classes = np.max(y_train) + 1
    # the fit is only needed to set up logreg
    logreg.fit(x_train[:n_classes], np.arange(n_classes))
    logreg.coef_ = np.mean(coefs, axis=0)
    logreg.intercept_ = np.mean(intercepts, axis=0)

    preds_test = logreg.predict(x_test)
    preds_train = logreg.predict(x_train)
    n_groups = np.max(g_train) + 1
    test_accs = [(preds_test == y_test)[g_test == g].mean()
                 for g in range(n_groups)]
    test_mean_acc = (preds_test == y_test).mean()
    train_accs = [(preds_train == y_train)[g_train == g].mean()
                  for g in range(n_groups)]
    return test_accs, test_mean_acc, train_accs



## Load data
if args.dataset == 'waterbirds':
    target_resolution = (224, 224)
    train_transform = get_transform_cub(target_resolution=target_resolution,
                                        train=True, augment_data=False)
    test_transform = get_transform_cub(target_resolution=target_resolution,
                                    train=False, augment_data=False)

    trainset = WaterBirdsDataset(
        basedir=args.data_dir, split="train", transform=train_transform)
    testset = WaterBirdsDataset(
        basedir=args.data_dir, split="test", transform=test_transform)
    valset = WaterBirdsDataset(
        basedir=args.data_dir, split="val", transform=test_transform)

    loader_kwargs = {'batch_size': args.batch_size,
                    'num_workers': 4, 'pin_memory': True,
                    "reweight_places": None}
    train_loader = get_loader(
        trainset, train=True, reweight_groups=False, reweight_classes=False,
        **loader_kwargs)
    test_loader = get_loader(
        testset, train=False, reweight_groups=None, reweight_classes=None,
        **loader_kwargs)
    val_loader = get_loader(
        valset, train=False, reweight_groups=None, reweight_classes=None,
        **loader_kwargs)
elif args.dataset == 'metashift':
    train_data, val_data, test_data = prepare_confounder_data(train=True, data_dir=args.data_dir)

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 1, 'pin_memory': False}
    train_loader = train_data.get_loader(reweight_groups=False,
                                                train=True, **loader_kwargs)

    test_loader = test_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)
    val_loader = val_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)
    trainset = train_data
    trainset.n_places = 2
elif args.dataset == 'domino':
    spuriosity = 90
    trainset = torch.load(os.path.join(args.data_dir, 'train_'+str(spuriosity)+'.pt'))
    valset = torch.load(os.path.join(args.data_dir, 'val_'+str(spuriosity)+'.pt'))
    testset = torch.load(os.path.join(args.data_dir, 'test_'+str(spuriosity)+'.pt'))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, drop_last = False, num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last = False, num_workers=1)

    trainset.n_classes = 2
    trainset.n_places = 2
    trainset.n_groups = 4
    valset.n_groups = 4
    testset.n_groups = 4
else:
    raise ValueError('Invalid dataset')


# Load model
def load_checkpoint(model, checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    else:
        state = torch.load(checkpoint_path)
        i = 0
        if isinstance(model, torch.nn.DataParallel):
            model_dict = model.module.state_dict()
        else:
            model_dict = model.state_dict()
        model_keys = list(model_dict.keys())
        state_temp = state
        if 'model' in state.keys():
            state_temp = state['model']
        for key in list(state_temp.keys()):
            if i < len(model_keys) and model_keys[i] in key:
                model_dict[model_keys[i]] = state_temp[key]
                i += 1
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(model_dict)
        else:
            model.load_state_dict(model_dict)
        del state
        torch.cuda.empty_cache()
        return model

n_classes = trainset.n_classes
model = torchvision.models.resnet50(pretrained=False)
d = model.fc.in_features
model.fc = torch.nn.Linear(d, n_classes)
# model.load_state_dict(torch.load(args.ckpt_path))
if args.dataset == 'metashift':
    model = torch.load(args.ckpt_path)
else:
    model = load_checkpoint(model, checkpoint_path=args.ckpt_path)
model.cuda()
model.eval()

# Evaluate model
print("Base Model")
base_model_results = {}
get_yp_func = partial(get_y_p, n_places=trainset.n_places)
base_model_results["test"] = evaluate(model, test_loader, get_yp_func)
base_model_results["val"] = evaluate(model, val_loader, get_yp_func)
base_model_results["train"] = evaluate(model, train_loader, get_yp_func)
print(base_model_results)
print()

model.eval()

# Extract embeddings
def get_embed(m, x):
    x = m.conv1(x)
    x = m.bn1(x)
    x = m.relu(x)
    x = m.maxpool(x)

    x = m.layer1(x)
    x = m.layer2(x)
    x = m.layer3(x)
    x = m.layer4(x)

    x = m.avgpool(x)
    x = torch.flatten(x, 1)
    return x


all_embeddings = {}
all_y, all_p, all_g = {}, {}, {}
for name, loader in [("train", train_loader), ("test", test_loader), ("val", val_loader)]:
    all_embeddings[name] = []
    all_y[name], all_p[name], all_g[name] = [], [], []
    for x, y, g, p in tqdm.tqdm(loader):
        with torch.no_grad():
            all_embeddings[name].append(get_embed(model, x.cuda()).detach().cpu().numpy())
            all_y[name].append(y.detach().cpu().numpy())
            all_g[name].append(g.detach().cpu().numpy())
            all_p[name].append(p.detach().cpu().numpy())
    all_embeddings[name] = np.vstack(all_embeddings[name])
    all_y[name] = np.concatenate(all_y[name])
    all_g[name] = np.concatenate(all_g[name])
    all_p[name] = np.concatenate(all_p[name])


# DFR on validation
print("DFR on validation")
dfr_val_results = {}
c, w1, w2 = dfr_on_validation_tune(
    all_embeddings, all_y, all_g,
    balance_val=args.balance_dfr_val, add_train=not args.notrain_dfr_val)
dfr_val_results["best_hypers"] = (c, w1, w2)
print("Hypers:", (c, w1, w2))
test_accs, test_mean_acc, train_accs = dfr_on_validation_eval(
        c, w1, w2, all_embeddings, all_y, all_g,
    balance_val=args.balance_dfr_val, add_train=not args.notrain_dfr_val)
dfr_val_results["test_accs"] = test_accs
dfr_val_results["train_accs"] = train_accs
dfr_val_results["test_worst_acc"] = np.min(test_accs)
dfr_val_results["test_mean_acc"] = test_mean_acc
print(dfr_val_results)
print()

worst_gacc = dfr_val_results["test_worst_acc"]
mean_gacc = test_mean_acc

# DFR on train subsampled
'''print("DFR on train subsampled")
dfr_train_results = {}
c, w1, w2 = dfr_train_subset_tune(
    all_embeddings, all_y, all_g,
    learn_class_weights=args.tune_class_weights_dfr_train)
dfr_train_results["best_hypers"] = (c, w1, w2)
print("Hypers:", (c, w1, w2))
test_accs, test_mean_acc, train_accs = dfr_train_subset_eval(
        c, w1, w2, all_embeddings, all_y, all_g)
dfr_train_results["test_accs"] = test_accs
dfr_train_results["train_accs"] = train_accs
dfr_train_results["test_worst_acc"] = np.min(test_accs)
dfr_train_results["test_mean_acc"] = test_mean_acc
print(dfr_train_results)
print()


all_results = {}
all_results["base_model_results"] = base_model_results
all_results["dfr_val_results"] = dfr_val_results
all_results["dfr_train_results"] = dfr_train_results
print(all_results)

with open(args.result_path, 'wb') as f:
    pickle.dump(all_results, f)
'''

print(SEED, worst_gacc, mean_gacc)

fp = open('res.txt', 'w')
fp.write(f'{SEED} {worst_gacc} {mean_gacc}')
fp.close()
