import os
import json
import argparse
import torch
import random
import torch.optim.lr_scheduler
from models import gresnet32_2, gresnet18, gresnet18mlp
from mind import MIND
from copy import deepcopy
from utils.generic import freeze_model, set_seed, setup_logger
from utils.publisher import push_results
from utils.transforms import to_tensor_and_normalize, default_transforms,default_transforms_core50,\
    to_tensor_and_normalize_core50,default_transforms_TinyImageNet,to_tensor_and_normalize_TinyImageNet, default_transforms_Synbols,to_tensor_and_normalize_Synbols, to_tensor
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from continuum import ClassIncremental
from continuum.tasks import split_train_val
from continuum.datasets import CIFAR100,Core50
from test_fn import test#, pert_CSI
import pickle as pkl
from parse import args
from utils.core50dset import get_all_core50_data, get_all_core50_scenario
from utils.tiny_imagenet_dset import get_all_tinyImageNet_data
from utils.synbols_dset import get_synbols_data
from continuum.datasets import InMemoryDataset
from continuum.scenarios import ContinualScenario
import numpy as np
from time import time
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torchvision.transforms as transforms
import pingouin as pg


to_tensor_and_normalize_2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        ),
    ])

default_transforms_2 = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=1.0),
        #transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        ),
    ])

def get_stat_exp(y, y_hats, exp_idx, task_id, task_predictions):

    """ Compute accuracy and task accuracy for each experience."""
    conf_mat = torch.zeros((exp_idx+1, exp_idx+1))
    for i in range(exp_idx+1):
        ybuff= y[task_id==i]
        y_hats_buff=y_hats[task_id==i]
        acc = (ybuff==y_hats_buff).sum()/y_hats_buff.shape[0]

        for j in range(exp_idx+1):
            conf_mat[i,j] = ((task_id==i)&(task_predictions==j)).sum()/(task_id==i).sum()

        print(f"EXP:{i}, acc:{acc:.3f}, task:{conf_mat[i,i]:.3f}, distrib:{[round(conf_mat[i,j].item(), 3) for j in range(exp_idx+1)]}")


def entropy(vec):
    return -torch.sum(vec * torch.log(vec + 1e-7), dim=1)

# model
if args.model == 'gresnet32':
    model = gresnet32_2(dropout_rate = args.dropout)
elif args.model == 'gresnet18':
    model = gresnet18(num_classes=args.n_classes)
elif args.model == 'gresnet18mlp':
    model = gresnet18mlp(num_classes=args.n_classes)
else:
    raise ValueError("Model not found.")

data_path = os.path.expanduser('/davinci-1/home/dmor/PycharmProjects/MIND/data_64x64')

# log files
setup_logger()

acc_ = []
num_exp = 101
for ss in range(0,1):
    set_seed(ss)
    file_name = f"cifar100_baseline_{ss}"
    model.load_state_dict(torch.load(f"/davinci-1/home/dmor/PycharmProjects/MIND/logs/{file_name}/checkpoints/weights.pt"))
    bn_weights = pkl.load(open(f"/davinci-1/home/dmor/PycharmProjects/MIND/logs/{file_name}/checkpoints/bn_weights.pkl", "rb"))
    model.bn_weights = bn_weights

    model.to(args.device)

    strategy = MIND(model)

    if args.dataset == 'CIFAR100':
        class_order = list(range(100))
        random.shuffle(class_order)

        train_dataset = CIFAR100(data_path, download=True, train=True)
        test_dataset = CIFAR100(data_path, download=True, train=False)

        x = train_dataset.get_data()[0]
        y = train_dataset.get_data()[1]

        r = int(args.extra_classes / args.classes_per_exp)

        if args.mode == 3:
            # modifico i dati in se (train)
            new_y = []
            new_x = []
            old_x = train_dataset.get_data()[0]
            old_y = train_dataset.get_data()[1]
            for k in range(r):
                for i in range(len(old_y)):
                    new_y.append(old_y[i])
                    new_y.append(old_y[i])
                    new_x.append(old_x[i])
                    new_x.append(np.rot90(old_x[i], k+1))
            new_x = np.array(new_x)
            new_y = np.array(new_y)
            train_dataset = InMemoryDataset(new_x, new_y)

        if args.mode == 4:
            # modifico il class order
            class_order_ = []
            for t in range(10):
                for k in range(r+1):
                    for c in range(10):
                        class_order_.append(class_order[t * 10 + c]+100*k)
            class_order = class_order_

            # modifico i dati in se (train)

            new_y = []
            new_x = []
            old_x = train_dataset.get_data()[0]
            old_y = train_dataset.get_data()[1]
            for i in range(len(old_y)):
                new_y.append(old_y[i])
                new_x.append(old_x[i])
                new_y.append(old_y[i]+100)
                new_x.append(old_x[i][:, :, [2, 1, 0]])
            new_x = np.array(new_x)
            new_y = np.array(new_y)
            train_dataset = InMemoryDataset(new_x, new_y)

            # modifico i dati in se (test)
            new_y = []
            new_x = []
            old_x = test_dataset.get_data()[0]
            old_y = test_dataset.get_data()[1]

            for i in range(len(old_y)):
                new_y.append(old_y[i])
                new_y.append(old_y[i] + 100)
                new_x.append(old_x[i])
                new_x.append(np.rot90(old_x[i], 2))
            new_x = np.array(new_x)
            new_y = np.array(new_y)
            test_dataset = InMemoryDataset(new_x, new_y)

            strategy.train_scenario = ClassIncremental(
                train_dataset,
                increment=args.classes_per_exp + args.extra_classes,
                class_order=class_order,
                transformations=default_transforms)
        else:
            strategy.train_scenario = ClassIncremental(
                train_dataset,
                increment=args.classes_per_exp,
                class_order=class_order,
                transformations=default_transforms)

            strategy.train_scenario_2 = ClassIncremental(
                train_dataset,
                increment=args.classes_per_exp,
                class_order=class_order,
                transformations=to_tensor)

        if args.mode == 4:
            inc = args.classes_per_exp + args.extra_classes
        else:
            inc = args.classes_per_exp

        #if args.aug_inf:
        #    tra = to_tensor
        #else:
        #    tra = to_tensor_and_normalize

        tra = to_tensor

        strategy.test_scenario = ClassIncremental(
            test_dataset,
            increment=inc,
            class_order=class_order,
            transformations=tra)
        strategy.test_scenario_2 = ClassIncremental(
            test_dataset,
            increment=inc,
            class_order=class_order,
            transformations=to_tensor_and_normalize)



    print(f"Number of classes: {strategy.train_scenario.nb_classes}.")
    print(f"Number of tasks: {strategy.train_scenario.nb_tasks}.")

    strategy.pruner.masks = torch.load(f"/davinci-1/home/dmor/PycharmProjects/MIND/logs/{file_name}/checkpoints/masks.pt")

    for i, train_taskset in enumerate(strategy.train_scenario):
        if args.packnet_original:
            with torch.no_grad():
                strategy.pruner.dezero(strategy.model)

        strategy.experience_idx = i
        strategy.model.set_output_mask(i, train_taskset.get_classes())

        model.load_bn_params(strategy.experience_idx)

        # prepare dataset
        strategy.train_taskset, strategy.val_taskset = split_train_val(train_taskset, val_split=args.val_split)
        strategy.train_dataloader = DataLoader(strategy.train_taskset, batch_size=args.bsize, shuffle=True)
        if len(strategy.val_taskset):
            strategy.val_dataloader = DataLoader(strategy.val_taskset, batch_size=args.bsize, shuffle=True)
        else:
            strategy.val_dataloader = DataLoader(strategy.test_scenario_2[i], batch_size=args.bsize, shuffle=True)

        #################### TEST ##########################
        if i != 9:
            continue

        with torch.no_grad():
            acc = []
            acc_e = []
            taw = []
            cl = []

            test_set=strategy.test_scenario[:i+1]
            strategy.model.eval()
            dataloader = DataLoader(strategy.train_scenario[:i+1], batch_size=1000, shuffle=False, num_workers=8)
            dataloader_2 = DataLoader(strategy.train_scenario_2[:i + 1], batch_size=1000, shuffle=False, num_workers=8)

            hist_1 = []
            hist_2 = []

            for i, (x, y, task_id) in enumerate(dataloader):
                for j in range(strategy.experience_idx + 1):

                    # create a temporary model copy
                    model = freeze_model(deepcopy(strategy.model))

                    strategy.pruner.set_gating_masks(model, j, weight_sharing=args.weight_sharing, distillation=True)
                    model.load_bn_params(j)
                    model.exp_idx = j

                    if j==9:
                        pred = model(x.to(args.device)).squeeze()

                        hist_1.append(pred)
                        hist_2.append(y)

            '''imgs = [0]*100
            for i, (x, y, task_id) in enumerate(dataloader_2):
                for k in range(len(task_id)):
                    imgs[y[k]] = x[k]

            scores = []
            for img in imgs:
                img_ = [transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))(torch.tensor(img).clone())]
                for p in range(100):
                    img_.append(default_transforms_2(img.clone()))
                img_ = torch.stack(img_)
                out = model(img_.to(args.device))
                dist = torch.norm(out[1:] - out[0], dim=1)
                scores.append(dist.mean().item())'''

            matrix = np.zeros((100, 10))
            matrix_2 = np.zeros((100, 10))

            for j in range(strategy.experience_idx + 1):
                print(j)
                # create a temporary model copy
                model = freeze_model(deepcopy(strategy.model))
                strategy.pruner.set_gating_masks(model, j, weight_sharing=args.weight_sharing, distillation=True)
                model.load_bn_params(j)
                model.exp_idx = j

                scores = [0]*100
                scores_2 = [0] * 100
                samples = [0]*100
                for i, (x, y, task_id) in enumerate(dataloader_2):
                    for k in range(len(task_id)):
                        if samples[y[k]]<500:
                            img = x[k]
                            img_ = [transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))(
                                torch.tensor(img).clone())]
                            '''for p in range(1):
                                img_.append(default_transforms_2(img.clone()))'''
                            img_ = torch.stack(img_)
                            out = model(img_.to(args.device))
                            '''dist = torch.norm(out[1:] - out[0], dim=1)
                            if samples[y[k]]<250:
                                scores[y[k]] += dist.mean().item()
                                samples[y[k]] += 1
                            else:
                                scores_2[y[k]] += dist.mean().item()
                                samples[y[k]] += 1'''

                '''result = [x / y for x, y in zip(scores, samples)]
                result_2 = [x / y for x, y in zip(scores_2, samples)]

                for i in range(100):
                    print(result[i]/sum(result))
                    print(result_2[i] / sum(result_2))
                    print('\n')
                    matrix[i][j] = result[i]/sum(result)
                    matrix_2[i][j] = result_2[i] / sum(result_2)'''

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            axs[0].imshow(matrix, cmap='viridis', interpolation='nearest')
            axs[0].set_title("Immagine 1")
            axs[0].axis('off')

            axs[1].imshow(matrix_2, cmap='viridis', interpolation='nearest')
            axs[1].set_title("Immagine 2")
            axs[1].axis('off')
            plt.show()

            y=torch.tensor(y)

            hist_1.append(out)
            hist_2.append(y)




            X = torch.cat(hist_1, dim=0)
            labels = torch.cat(hist_2, dim=0)

            # Riduzione della dimensionalità con TSNE
            exp=9
            X = torch.cat([X[(labels<(10*(exp+1)))&(labels>=(exp*10)),:], X[labels<0,:]])
            labels = torch.cat([labels[((labels<(10*(exp+1)))&(labels>=(exp*10)))],labels[labels<0]])


            tsne = TSNE(n_components=2, perplexity=10, random_state=42)
            X_2d = tsne.fit_transform(X.cpu().numpy())


            # Primo grafico per labels < 10
            plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels.cpu().numpy()[:], cmap='tab10', alpha=0.7, marker='o')
            plt.title("Classi Originali")
            plt.xlabel("Dimensione 1")
            plt.ylabel("Dimensione 2")

            # Secondo grafico per labels >= 10
            plt.subplot(1, 2, 2)
            mask = labels >= 10
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], c=labels.cpu().numpy()[mask], cmap='tab10', alpha=0.7, marker='o')
            plt.title("Classi ruotate")
            plt.xlabel("Dimensione 1")
            plt.ylabel("Dimensione 2")

            plt.tight_layout()
            plt.show()