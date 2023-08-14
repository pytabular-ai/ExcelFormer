import os
import math
import time
import json
import random
import argparse
import numpy as np
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from category_encoders import CatBoostEncoder


from ver1.bin import (
    ExcelFormerV, ExcelFormerA
)
from ver1.lib import Transformations, build_dataset, prepare_tensors, DATA, make_optimizer, PROJ


IMPLEMENTED_MODELS = [
    ExcelFormerV, ExcelFormerA
]

YANDEX_DATASETS = [
    'california', 'churn', 'gesture', 'helena', 'house', 
    'jannis', 'adult', 'eye', 'higgs-small', 'otto', 'fb-comments', 'covtype']
BENCHMARK_DATASETS = [
    'Ailerons', 'Aileronsv2', 'analcatdata_supreme', 'bank-marketing', 'Bike_Sharing_Demand', 
    'black_friday', 'Brazilian_houses', 'compass', 'cpu_act', 'credit', 'diamonds', 'electricity', 
    'elevators', 'elevatorsv2', 'fifa', 'house_sales', 'MagicTelescope', 'medical_charges', 'MiamiHousing2016', 
    'nyc-taxi-green-dec-2016', 'OnlineNewsPopularity', 'particulate-matter-ukair-2017', 'phoneme', 'pol', 'polv2', 
    'rl', 'road-safety', 'SGEMM_GPU_kernel_performance', 'sulfur', 'superconduct', 
    'visualizing_soil', 'wine_quality', 'yprop_4_1', 'isolet']


def get_training_args():
    MODEL_CARDS = [x.__name__ for x in IMPLEMENTED_MODELS]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='result_excel_single_beta_tuned')
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--normalization", type=str, default='quantile')
    parser.add_argument("--model", type=str, choices=MODEL_CARDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_prenorm", action='store_true')
    # parser.add_argument("--early_stop", type=int, default=10)
    args = parser.parse_args()
    # store_args = vars(args)
    
    
    args.output = args.output + f'/{args.model}/{args.dataset}/{args.seed}'
    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    # default config
    cfg = {
        "model": {
            "activation": "reglu", 
            "initialization": "kaiming", 
            "prenormalization": True
        },
        "training": {
            "max_epoch": 500,
            "optimizer": "adamw",
        }
    }

    # tuned config
    cfg_file = f'configs_excel_single_beta_tune/{args.dataset}/{args.model}/cfg.json'
    assert os.path.exists(cfg_file)
    with open(cfg_file, 'r') as f:
        tuned_cfg = json.load(f)
    cfg['model'].update(tuned_cfg['model'])
    cfg['training'].update(tuned_cfg['training'])
    cfg['mixup'] = tuned_cfg['mixup']
    if args.no_prenorm:
        cfg['model']['prenormalization'] = False
    
    return args, cfg


def seed_everything(seed=42):
    '''
    Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.
    '''
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



"""args"""
device = torch.device('cuda')
args, cfg = get_training_args()
seed_everything(args.seed)

"""Datasets and Dataloaders"""
if args.dataset in YANDEX_DATASETS:
    from ver1.lib import YANDEX_DATA as DATA
elif args.dataset in BENCHMARK_DATASETS:
    from ver1.lib import BENCHMARK_DATA as DATA

dataset_name = args.dataset
T_cache = True
normalization = args.normalization if args.normalization != '__none__' else None
transformation = Transformations(normalization=normalization)
dataset = build_dataset(DATA / dataset_name, transformation, T_cache)

"""CatBoost Encoder"""
if dataset.X_num['train'].dtype == np.float64:
    dataset.X_num = {k: v.astype(np.float32) for k, v in dataset.X_num.items()}
if dataset.X_cat is not None:
    cardinalities = dataset.get_category_sizes('train')
    enc = CatBoostEncoder(
        cols=list(range(len(cardinalities))), 
        return_df=False
    ).fit(dataset.X_cat['train'], dataset.y['train'])
    for k in ['train', 'val', 'test']:
        # 1: directly regard catgorical features as numerical
        dataset.X_num[k] = np.concatenate([enc.transform(dataset.X_cat[k]).astype(np.float32), dataset.X_num[k]], axis=1)


d_out = dataset.n_classes or 1
X_num, X_cat, ys = prepare_tensors(dataset, device=device)
X_cat = None # current categorical encoding

###################### sort numerical features with MMI ###########################
mmi_cache_file = f'cache/mmi/{args.dataset}.npy'
if os.path.exists(mmi_cache_file):
    mi_scores = np.load(mmi_cache_file)
else:
    mi_func = mutual_info_regression if dataset.is_regression else mutual_info_classif
    mi_scores = mi_func(dataset.X_num['train'], dataset.y['train'])
    np.save(mmi_cache_file, mi_scores)
mi_ranks = np.argsort(-mi_scores)
X_num = {k: v[:, mi_ranks] for k, v in X_num.items()}
sorted_mi_scores = torch.from_numpy(mi_scores[mi_ranks] / mi_scores.sum()).float().to(device)

if dataset.task_type.value == 'regression':
    y_std = ys['train'].std().item()

"""dataset argument"""
args.early_stop = 200

batch_size_dict = {
    'churn': 128, 'eye': 128, 'gesture': 128, 'california': 256, 'house': 256, 'adult': 256 , 
    'higgs-small': 512, 'helena': 512, 'jannis': 512, 'otto': 512, 'fb-comments': 512,
    'covtype': 1024, 'year': 1024, 'santander': 1024, 'microsoft': 1024, 'yahoo': 256}
if args.dataset == 'epsilon':
    batch_size = 16 if args.dataset == 'epsilon' else 128 if args.dataset == 'yahoo' else 256
elif args.dataset in batch_size_dict:
    # Yandex Dataset
    batch_size = batch_size_dict[args.dataset]
    val_batch_size = 512
else:
    if dataset.n_features <= 32:
        batch_size = 512
        val_batch_size = 8192
    elif dataset.n_features <= 100:
        batch_size = 128
        val_batch_size = 512
    elif dataset.n_features <= 1000:
        batch_size = 32
        val_batch_size = 64
    else:
        batch_size = 16
        val_batch_size = 16

cfg['training'].update({
    "batch_size": batch_size, 
    "eval_batch_size": val_batch_size, 
    "patience": args.early_stop
})


num_workers = 0
data_list = [X_num, ys] if X_cat is None else [X_num, X_cat, ys]
train_dataset = TensorDataset(*(d['train'] for d in data_list))
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)
val_dataset = TensorDataset(*(d['val'] for d in data_list))
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=num_workers
)
test_dataset = TensorDataset(*(d['test'] for d in data_list))
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=num_workers
)
dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}


"""Loss Function"""
loss_fn = (
    F.binary_cross_entropy_with_logits
    if dataset.is_binclass
    else F.cross_entropy
    if dataset.is_multiclass
    else F.mse_loss
)

"""utils function"""
def apply_model(model, x_num, x_cat, mixup=True, beta=None, mtype:str=None):
    if any(issubclass(eval(args.model), x) for x in IMPLEMENTED_MODELS):
        use_mixup = mixup and model.training
        return model(x_num, x_cat, mixup=use_mixup, beta=beta, mtype=mtype)
    else:
        raise NotImplementedError

@torch.inference_mode()
def evaluate(model, parts):
    model.eval()
    predictions = {}
    for part in parts:
        assert part in ['train', 'val', 'test']
        predictions[part] = []
        for batch in dataloaders[part]:
            x_num, x_cat, y = (
                (batch[0], None, batch[1])
                if len(batch) == 2
                else batch
            )
            predictions[part].append(apply_model(model, x_num, x_cat, mixup=False))
        predictions[part] = torch.cat(predictions[part]).cpu().numpy()
    prediction_type = None if dataset.is_regression else 'logits'
    return dataset.calculate_metrics(predictions, prediction_type)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_predict(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    beta: float, 
    mtype:str, 
    metric: str
):
    init_score = evaluate(model, ['test'])['test'][metric]
    print(f'Test score before training: {init_score: .4f}')
    """info containers"""
    losses, val_metric, test_metric = [], [], []
    loss_holder = AverageMeter()
    best_score = -np.inf # best val score
    final_test_score = -np.inf
    best_test_score = -np.inf
    no_improvement = 0
    best_epoch = 0 # epoch of best val score
    # print info
    report_frequency = len(ys['train']) // batch_size // 2
    """Training Args"""
    n_epochs = cfg['training']['max_epoch']
    EARLY_STOP = cfg['training']['patience']

    # default: warmup and lr scheduler
    warm_up = 10 # warm up epoch
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=n_epochs - warm_up) # lr decay
    max_lr = cfg['training']['lr']
        

    for epoch in range(1, n_epochs + 1):
        model.train()
        if warm_up > 0 and epoch <= warm_up:
            # warm up lr
            lr = max_lr * epoch / warm_up
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # lr decay
            scheduler.step()
        
        for iteration, batch in enumerate(train_loader):
            x_num, x_cat, y = (
                (batch[0], None, batch[1]) # default: CatBoost Encoder
                if len(batch) == 2
                else batch
            )
            optimizer.zero_grad()
            if mtype == 'none':
                loss = loss_fn(apply_model(model, x_num, x_cat, mixup=False), y)
            else:
                preds, feat_masks, shuffled_ids = apply_model(model, x_num, x_cat, mixup=True, beta=beta, mtype=mtype)
                if mtype == 'feat_wise':
                    lambdas = (sorted_mi_scores * feat_masks).sum(1) # bs
                    lambdas2 = 1 - lambdas
                elif mtype == 'dim_wise':
                    lambdas = feat_masks
                    lambdas2 = 1 - lambdas
                if dataset.is_regression:
                    mix_y = lambdas * y + lambdas2 * y[shuffled_ids]
                    loss = loss_fn(preds, mix_y)
                else:
                    loss = lambdas * loss_fn(preds, y, reduction='none') + lambdas2 * loss_fn(preds, y[shuffled_ids], reduction='none')
                    loss = loss.mean()
            loss.backward()
            optimizer.step()
            loss_holder.update(loss.item(), len(ys))
            if iteration % report_frequency == 0:
                print(f'(epoch) {epoch} (batch) {iteration} (loss) {loss_holder.val:.4f} (avg_loss) {loss_holder.avg:.4f}')
        
        losses.append(loss_holder.avg)
        loss_holder.reset() # clear losses for next epoch
        
        scores = evaluate(model, ['val', 'test'])
        val_score, test_score = scores['val'][metric], scores['test'][metric]
        val_metric.append(val_score), test_metric.append(test_score)
        print(f'Epoch {epoch:03d} | Validation score: {val_score:.4f} | Test score: {test_score:.4f}', end='')

        # if dataset.is_binclass:
        #     val_score = scores['val']['roc_auc']
        # else:
        #     val_score = scores['val']['score']
        if val_score > best_score:
            best_score = val_score
            final_test_score = test_score
            best_epoch = epoch
            print(' <<< BEST VALIDATION EPOCH')
            no_improvement = 0
        else:
            no_improvement += 1
        # record best test socre (to check overfitting)
        if test_score > best_test_score:
            best_test_score = test_score
        if no_improvement == EARLY_STOP:
            break
    return {
        'epoch': best_epoch,
        'metric': (
            'rmse'
            if dataset.is_regression
            else 'roc_auc'
            if dataset.is_binclass
            else 'accuracy'
        ),
        'best_val_score': best_score,
        'final': final_test_score,
        'best': best_test_score,
        'losses': str(losses),
        'val_score': str(val_metric),
        'test_score': str(test_metric),
    }


"""Models"""
model_cls = eval(args.model)
n_num_features = dataset.n_num_features # drop some features
cardinalities = None # default: CatBoost Encoder

# default settings
cfg['model'].setdefault('kv_compression', None)
cfg['model'].setdefault('kv_compression_sharing', None)
cfg['model'].setdefault('token_bias', True)

model_config = {
    'd_numerical': n_num_features,
    'd_out': d_out,
    'categories': cardinalities,
    **cfg['model']
}
model = model_cls(**model_config).to(device)
if torch.cuda.device_count() > 1:  # type: ignore[code]
    print('Using nn.DataParallel')
    model = nn.DataParallel(model)

"""Optimizers"""
def needs_wd(name):
    return all(x not in name for x in ['tokenizer', '.norm', '.bias'])
parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
optimizer = make_optimizer(
    cfg['training']['optimizer'],
    (
        [
            {'params': parameters_with_wd},
            {'params': parameters_without_wd, 'weight_decay': 0.0},
        ]
    ),
    cfg['training']['lr'],
    cfg['training']['weight_decay'],
)

"""Training and Predict"""
metric = 'roc_auc' if dataset.is_binclass else 'score' # using AUC for binary classification
results = train_predict(model, optimizer, cfg['mixup']['beta'], cfg['mixup']['mix_type'], metric)

"""Record Experiment Results"""
results = {
    'config': {
        'output': args.output,
        'dataset': args.dataset,
        'normalization': args.normalization,
        'seed': args.seed,
    },
    **results,
    'model': {'name': args.model, **cfg['model']},
    'training': cfg['training'],
    'mixup': cfg['mixup'],
}
exp_list = exp_list = [file for file in os.listdir(args.output) if '.json' in file]
exp_list = [int(file.split('.')[0]) for file in exp_list] # count of exp under this random seed
exp_id = 0 if len(exp_list) == 0 else max(exp_list) + 1
with open(f"{args.output}/{exp_id}.json", 'w', encoding='utf8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
