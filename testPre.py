import os
import os.path as osp
import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

from utils.parameters import get_parser
from utils.dataset import MyBlockChain_TUDataset
from utils.transform import *
from utils.tools import setup_seed, EarlyStopping, data_split
from utils.transform import Augmentor_Transform, MyAug_Identity

from model.encoder import *
from model.model import Ethident

# data information

label_abbreviation = {"i": "ico-wallets",
                      "m": "mining",
                      "e": "exchange",
                      "p": "phish-hack",
                      "r": "robot"}

args = get_parser()
setup_seed(args.seed)
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

label = label_abbreviation[args.label]  # target account label

# path
data_path = osp.join(osp.dirname(osp.realpath(__file__)), f'{args.root}',
                     '{}/{}/{}hop-{}/{}'.format(args.dataType, label, args.hop, args.topk, args.edge_sample_strategy))

view_1, view_2 = args.aug.split('+')
sample_scheme = ['Volume', 'Times', 'averVolume']
transform_raw = T.Compose([
    MyToUndirected(edge_attr_keys=['edge_attr']) if args.to_undirected else MyAug_Identity(),
    ColumnNormalizeFeatures(['edge_attr']),
    T.NormalizeFeatures()
])
dataset_raw = MyBlockChain_TUDataset(root=data_path, name=args.dataType.upper() + 'G',
                                     use_node_attr=args.use_node_attribute,
                                     use_node_importance=args.use_node_labeling,
                                     use_edge_attr=args.use_edge_attribute,  # feature selection
                                     transform=transform_raw)  # .shuffle()
train_splits, val_splits, test_splits = data_split(X=np.arange(len(dataset_raw)),
                                                   Y=np.array([dataset_raw[i].y.item() for i in range(len(dataset_raw))]),  # 等官方修复bug
                                                   # Y=np.array([dataset[i].y[0].item() for i in range(len(dataset))]),
                                                   seeds=args.seeds[:args.exp_num], K=args.k_ford)

if view_1 not in sample_scheme:
    transform_view_1 = T.Compose([
        Augmentor_Transform[view_1](prob=args.aug_prob1),
        MyToUndirected(edge_attr_keys=['edge_attr']) if args.to_undirected else MyAug_Identity(),
        ColumnNormalizeFeatures(['edge_attr']),
        T.NormalizeFeatures()
    ])
    dataset_v1 = MyBlockChain_TUDataset(root=data_path, name=args.dataType.upper() + 'G',
                                        use_node_attr=args.use_node_attribute,
                                        use_node_importance=args.use_node_labeling,
                                        use_edge_attr=args.use_edge_attribute,  # feature selection
                                        transform=transform_view_1)  # .shuffle()
else:
    transform_view_1 = T.Compose([
        MyToUndirected(edge_attr_keys=['edge_attr']) if args.to_undirected else MyAug_Identity(),
        ColumnNormalizeFeatures(['edge_attr']),
        T.NormalizeFeatures()
    ])
    data_path_v1 = osp.join(osp.dirname(osp.realpath(__file__)), f'{args.root}',
                            '{}/{}/{}hop-{}/{}'.format(args.dataType, label, args.hop, args.topk, view_1))
    dataset_v1 = MyBlockChain_TUDataset(root=data_path_v1, name=args.dataType.upper() + 'G',
                                        use_node_attr=args.use_node_attribute,
                                        use_node_importance=args.use_node_labeling,
                                        use_edge_attr=args.use_edge_attribute,  # feature selection
                                        transform=transform_view_1)  # .shuffle()

if view_2 not in sample_scheme:
    transform_view_2 = T.Compose([
        Augmentor_Transform[view_2](prob=args.aug_prob1),
        MyToUndirected(edge_attr_keys=['edge_attr']) if args.to_undirected else MyAug_Identity(),
        ColumnNormalizeFeatures(['edge_attr']),
        T.NormalizeFeatures()
    ])
    dataset_v2 = MyBlockChain_TUDataset(root=data_path, name=args.dataType.upper() + 'G',
                                        use_node_attr=args.use_node_attribute,
                                        use_node_importance=args.use_node_labeling,
                                        use_edge_attr=args.use_edge_attribute,  # feature selection
                                        transform=transform_view_2)  # .shuffle()
else:
    transform_view_2 = T.Compose([
        MyToUndirected(edge_attr_keys=['edge_attr']) if args.to_undirected else MyAug_Identity(),
        ColumnNormalizeFeatures(['edge_attr']),
        T.NormalizeFeatures()
    ])
    data_path_v2 = osp.join(osp.dirname(osp.realpath(__file__)), f'{args.root}',
                            '{}/{}/{}hop-{}/{}'.format(args.dataType, label, args.hop, args.topk, view_2))

    dataset_v2 = MyBlockChain_TUDataset(root=data_path_v2, name=args.dataType.upper() + 'G',
                                        use_node_attr=args.use_node_attribute,
                                        use_node_importance=args.use_node_labeling,
                                        use_edge_attr=args.use_edge_attribute,  # feature selection
                                        transform=transform_view_2)  # .shuffle()

    
val_dataset = dataset_raw[val_splits[0]]
test_dataset = dataset_raw[test_splits[0]]
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
f1_list = []



print('Loading val data ... ')
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
print('Loading test data ... ')
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

encoder = HGATE_encoder(dataset_raw.num_features, args.hidden_dim, out_channels=dataset_raw.num_classes,
                        edge_dim=dataset_raw.num_edge_features, num_layers=args.num_layers, pooling=args.pooling, dropout=args.dropout,
                        add_self_loops=True, use_edge_atten=False).to(device)

model = Ethident(in_channels=dataset_raw.num_features, hidden_channels=args.hidden_dim, out_channels=dataset_raw.num_classes, encoder=encoder,
                 num_layers=args.num_layers, pooling=args.pooling, dropout=args.dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
early_stopping = EarlyStopping(patience=20, min_delta=args.early_stop_mindelta)

@torch.no_grad()
def test(loader):
    model.eval()
    model_path = 'final_model_dict.pth'
    model.load_state_dict(torch.load(model_path))
   
    
    
    total_correct, total_loss = 0, 0
    y_pred_label_list = []
    y_true_label_list = []
    predictions = []

    for data in loader:
        data = data.to(device)
        node_reps, graph_reps, pred_out = model(data.x, data.edge_index, data.batch, data.edge_attr)
        
        loss = model.loss_su(pred_out, data.y)
        total_loss += float(loss) * data.num_graphs

        pred_out = F.softmax(pred_out, dim=1)
        pred = torch.argmax(pred_out, dim=1)

        y_pred_label_list.append(pred)
        y_true_label_list.append(data.y)
        predictions.append(pred.cpu().numpy())

    y_pred = torch.cat(y_pred_label_list).cpu().detach().numpy()
    y_true = torch.cat(y_true_label_list).cpu().detach().numpy()

    acc = f1_score(y_true=y_true, y_pred=y_pred, average='micro')

    return acc, total_loss / len(loader.dataset), predictions

for epoch in range(1, args.epochs):
    val_acc, val_loss, val_predictions = test(val_loader)
    test_acc, _, test_predictions = test(test_loader)
    
    print(f'Epoch {epoch}:')
    print(f'Val Predictions: {val_predictions}')
    print(f'Test Predictions: {test_predictions}')

    if args.early_stop:
        early_stopping(val_loss, results=[epoch, val_loss, val_acc, test_acc])
        if early_stopping.early_stop:
            print('\n=====final results=====')
            _epoch, _val_loss, _val_acc, _test_acc = early_stopping.best_results
            f1_list.append(_test_acc)
            print(f'Exp: {1},  Epoch: {_epoch:03d},       '     
                  f'Val_Loss: {_val_loss:.4f},        ,        '
                  f'Val_Acc: {_val_acc:.4f},        '
                  f'Test_Acc: {_test_acc:.4f}\n\n')
            break
    else:
        f1_list.append(test_acc)

    print(f'Exp: {1},  Epoch: {epoch:03d},       '
          f'Val_Loss: {val_loss:.4f},        '
          f'Val_Acc: {val_acc:.4f},        '
          )

print('Result in terms of f1-score: {} ~ {}\n\n\n'.format(np.mean(f1_list), np.std(f1_list)))