import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import get_evaluation_results
from Dataloader import load_data
import copy
from model import CGmini


def train(args, device):

    features, features_dis, adj_hat_list, labels, idx_labeled, idx_unlabeled, idx_val = load_data(args, device=device)
    print('Labeled sample:', len(idx_labeled))
    num_classes = len(np.unique(labels))
    labels = labels.to(device)
    num_view = len(adj_hat_list)
    input_dims = features.shape[1]

    model = CGmini(input_dims, args.hidden_dims, num_view, num_classes, args.dropout,
                args.alpha, args.beta, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    begin_time = time.time()
    best_acc = 0
    best_f1 = 0
    train_acc_list = []
    train_f1_list = []
    val_acc_list = []
    val_f1_list = []
    with tqdm(total=args.num_epoch, desc="Training") as pbar:
        for epoch in range(args.num_epoch):
            model.train()
            z1, z2 = model(features, features_dis, adj_hat_list)

            optimizer.zero_grad()
            loss = caculate_loss(z1, z2, labels, idx_labeled, idx_unlabeled + idx_val, args)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                z1, z2 = model(features, features_dis, adj_hat_list)

                pred_labels1 = torch.argmax((z1 + z2) / 2, 1).cpu().detach().numpy()
                val_acc1, val_f11 = get_evaluation_results(labels.cpu().detach().numpy()[idx_val],pred_labels1[idx_val])

                val_acc_list.append(val_acc1)
                val_f1_list.append(val_f11)

                if val_f11 > best_f1:
                    best_acc = val_acc1
                    best_f1 = val_f11
                    best_model = copy.deepcopy(model)

                pbar.set_postfix({'Loss': '{:.6f}'.format(loss.item()),
                                  'val_acc': '{:.2f}'.format(val_acc1 * 100),
                                  'val_f1': '{:.2f}'.format(val_f11 * 100),
                                  'best_val_acc': '{:.2f}'.format(best_acc * 100),
                                  'best_val_f1': '{:.2f}'.format(best_f1 * 100)}
                                 )
                pbar.update(1)

    cost_time = time.time() - begin_time
    model.eval()
    z1, z2 = best_model(features, features_dis, adj_hat_list)
    pred_labels = torch.argmax((z1 + z2) / 2, 1).cpu().detach().numpy()

    acc, f1 = get_evaluation_results(labels.cpu().detach().numpy()[idx_unlabeled], pred_labels[idx_unlabeled])

    print("Evaluating the model")
    print("ACC: {:.2f}, F1: {:.2f},  Time:{:}".format(acc * 100, f1 * 100, cost_time))

    return acc * 100, f1 * 100, cost_time, val_acc_list, val_f1_list, train_acc_list, train_f1_list



def caculate_loss(pred1, pred2, labels, idx_labeled, idx_unlabeled, args):
    l1 = F.cross_entropy(pred1[idx_labeled], labels[idx_labeled])
    l2 = F.cross_entropy(pred2[idx_labeled], labels[idx_labeled])
    pred1 *= args.tau
    pred2 *= args.tau
    l3 = F.cross_entropy(pred1[idx_unlabeled], F.softmax(pred2, dim=1)[idx_unlabeled])
    l4 = F.cross_entropy(pred2[idx_unlabeled], F.softmax(pred1, dim=1)[idx_unlabeled])
    loss = args.fai * (l1 + l2) + (1 - args.fai) * (l3 + l4)

    return loss
