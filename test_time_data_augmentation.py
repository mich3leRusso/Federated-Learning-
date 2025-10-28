import os
import json
import argparse
import torch
import random
import torch.optim.lr_scheduler
from models import gresnet32, gresnet18, gresnet18mlp
from mind import MIND
from copy import deepcopy
from utils.generic import freeze_model, set_seed, setup_logger
from utils.publisher import push_results
from utils.transforms import ttda_cifar100, normalize_cifar100, ttda_core50, normalize_core50, ttda_TinyImageNet, normalize_TinyImageNet, ttda_Synbols, normalize_Synbols, to_tensor
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
from itertools import combinations
from explainability import run_explainability_tools, SVCCA_starter

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

project_path = '/davinci-1/home/dmor/PycharmProjects/Refactoring_MIND'
data_path = '/davinci-1/home/dmor/PycharmProjects/Refactoring_MIND/data'

model = gresnet32(dropout_rate=args.dropout)

# log files
setup_logger()

acc_ = []
taw_ = []
task_ = []
AA = []
BB = []
num_exp = args.n_aug  #perchè il primo valore è la baseline
for seed in range(args.seed+1):
    set_seed(seed)

    file_name = args.run_name[:-2]+f"_{seed}"
    print(file_name)
    model.load_state_dict(torch.load(project_path + f"/logs/{file_name}/checkpoints/weights.pt"))
    bn_weights = pkl.load(open(project_path + f"/logs/{file_name}/checkpoints/bn_weights.pkl", "rb"))
    model.bn_weights = bn_weights
    model.to(args.device)
    strategy = MIND(model)

    if args.dataset == 'CIFAR100':
        train_dataset = CIFAR100(data_path, download=True, train=True).get_data()
        test_dataset = CIFAR100(data_path, download=True, train=False).get_data()
        transform_0 = ttda_cifar100
        transform_1 = normalize_cifar100
    elif args.dataset == 'CORE50_CI':
        data_path_ = os.path.expanduser(data_path + '/core50_128x128')
        train_dataset, test_dataset = get_all_core50_data(data_path_, args.n_experiences, split=0.8)
        transform_0 = ttda_core50
        transform_1 = normalize_core50
    elif args.dataset == 'TinyImageNet':
        data_path_ = os.path.expanduser(data_path)
        train_dataset, test_dataset = get_all_tinyImageNet_data(data_path_, args.n_experiences)
        transform_0 = ttda_TinyImageNet
        transform_1 = normalize_TinyImageNet
    elif args.dataset == 'Synbols':
        train_data, test_data = get_synbols_data(data_path, n_tasks=args.n_experiences)
        train_dataset = InMemoryDataset(*train_data).get_data()
        test_dataset = InMemoryDataset(*test_data).get_data()
        transform_0 = ttda_Synbols
        transform_1 = normalize_Synbols
    transform_2 = to_tensor

    r = args.class_augmentation - 1

    if (args.dataset == 'CIFAR100') | (args.dataset == 'Synbols'):
        class_order = list(range(args.n_classes))
        random.shuffle(class_order)
        class_order_ = []
        for t in range(args.n_experiences):
            for k in range(r*(1-args.control) + 1):
                for c in range(args.classes_per_exp):
                    class_order_.append(class_order[t * args.classes_per_exp + c] + args.n_classes * k )
        class_order = class_order_

    # modifying train set
    new_y = []
    new_x = []
    old_x = train_dataset[0]
    old_y = train_dataset[1]
    for i in range(len(old_y)):
        for k in range(r + 1):
            new_y.append(old_y[i] + args.n_classes * k * (1-args.control))
            new_x.append(np.rot90(old_x[i], k))
            #if k == 0:
            #    new_x.append(np.rot90(old_x[i], k))
            #else:
            #    new_x.append(np.rot90(old_x[i], k+old_y[i] % 3))
    new_x = np.array(new_x)
    new_y = np.array(new_y)

    if (args.dataset == 'CORE50_CI') | (args.dataset == 'TinyImageNet'):
        new_z = []
        old_z = train_dataset[2]
        for i in range(old_x.shape[0]):
            for k in range(r + 1):
                new_z.append(old_z[i])
        new_z = np.array(new_z)
        class_order = []
        for i in range(args.n_experiences):
            classes_in_task = np.unique(new_y[new_z == i])
            for j in range(len(classes_in_task)):
                class_order.append(int(classes_in_task[j]))

    # modifying test set
    new_y = []
    new_x = []
    old_x = test_dataset[0]
    old_y = test_dataset[1]
    for i in range(args.n_experiences * args.extra_classes):
        new_y.append(args.n_classes + i)
        new_x.append(old_x[0])
    for i in range(len(old_y)):
        new_y.append(old_y[i])
        new_x.append(old_x[i])
    new_x = np.array(new_x)
    new_y = np.array(new_y)
    test_dataset = InMemoryDataset(new_x, new_y)

    strategy.test_scenario = ClassIncremental(
        test_dataset,
        increment=args.classes_per_exp + args.extra_classes,
        class_order=class_order,
        transformations=transform_2)

    strategy.pruner.masks = torch.load(project_path + f"/logs/{file_name}/checkpoints/masks.pt")

    for i, test_taskset in enumerate(strategy.test_scenario):
        if args.packnet_original:
            with torch.no_grad():
                strategy.pruner.dezero(strategy.model)

        strategy.experience_idx = i
        strategy.model.set_output_mask(i, test_taskset.get_classes())

        model.load_bn_params(strategy.experience_idx)

    with torch.no_grad():
        acc = []
        taw = []
        task = []

        test_set=strategy.test_scenario[:9+1]
        strategy.model.eval()
        dataloader = DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=8)

        hist_1 = []
        hist_2 = []
        hist_3 = []

        num_rot = args.class_augmentation

        if args.with_rotations == 0:
            num_rot = 1

        mean_t = []
        std_t = []
        mean_f = []
        std_f = []
        mean_e = []
        std_e = []
        for k in range(num_exp): #run sulle varie aumentazioni
            print(f"augmentation_number = {k}")
            if k==0:
                rot=0
            else:
                rot = k % num_rot

            s = args.n_classes + int(args.extra_classes * args.n_experiences)
            confusion_mat = torch.zeros((s, s))
            confusion_mat_taw = torch.zeros((s, s))

            #confusion_mat_aux = torch.zeros((int(s/args.class_augmentation), s))
            confusion_mat_aux = torch.zeros((s,s))

            y_hats = []
            y_taw = []
            y_aux = []
            ys = []
            task_ids = []
            task_predictions = []

            A = [] #true task prediction
            B = [] #false task prediction

            for i, (x, y, task_id) in enumerate(dataloader):
                frag_preds = []
                frag_preds_aux = []
                frag_preds_latent = []
                for j in range(strategy.experience_idx + 1):
                    # create a temporary model copy
                    model = freeze_model(deepcopy(strategy.model))

                    strategy.pruner.set_gating_masks(model, j, weight_sharing=args.weight_sharing, distillation=True)
                    model.load_bn_params(j)
                    model.exp_idx = j

                    if k > 1 + 2*(1-args.with_rotations)*(args.class_augmentation-1):
                        trans = transforms.Compose(transform_0)
                    else:
                        trans = transforms.Compose(transform_1)

                    if args.with_rotations == 1:
                        if (args.class_augmentation%2) == 0:
                            p = (k % args.class_augmentation + int(k / args.class_augmentation))%2
                        else:
                            p = k % 2
                    else:
                        p = k % 2

                    trans2 = transforms.RandomHorizontalFlip(p=p)

                    x_ = []
                    for img in x:
                        img = np.array(img)
                        if (args.dataset == "CORE50_CI") | (args.dataset == "Synbols"):
                            img = np.rot90(img.transpose(2, 1, 0), -rot).transpose(2, 1, 0)
                        else:
                            img = np.rot90(img.transpose(2, 1, 0), rot).transpose(2, 1, 0)
                        img = torch.tensor(img.copy())
                        new_img = trans(img)
                        new_img = trans2(new_img)
                        x_.append(new_img)
                    x_2 = torch.stack(x_)

                    pred = model(x_2.to(args.device))

                    pred = pred[:, j * (args.classes_per_exp + args.extra_classes): (j + 1) * (args.classes_per_exp + args.extra_classes)]

                    # removing scores associated with extra classes
                    if args.control_2 == 0:
                        #pass
                        sp = torch.softmax(pred / args.temperature, dim=1)
                        sp = sp[:, args.classes_per_exp*rot*(1-args.control):args.classes_per_exp*(rot*(1-args.control)+1)]
                    else:
                        sp = pred[:, args.classes_per_exp * rot:args.classes_per_exp * (rot + 1)]
                        sp = torch.softmax(sp / args.temperature, dim=1)

                    #frag_preds.append(torch.softmax(sp / args.temperature, dim=1))
                    frag_preds.append(sp)
                    frag_preds_aux.append(sp)
                    frag_preds_latent.append(torch.softmax(pred / args.temperature, dim=1))
                    #frag_preds_latent.append(pred)

                frag_preds = torch.stack(frag_preds)  # [n_frag, bsize, n_classes]
                frag_preds_aux = torch.stack(frag_preds_aux)
                frag_preds_latent = torch.stack(frag_preds_latent)

                task_id = task_id.long()
                n = frag_preds.shape[1]
                x_max, _ = frag_preds.max(dim=2)
                true = x_max[task_id, torch.arange(n)]
                all_indices = torch.arange(10).unsqueeze(1).expand(10, n)
                mask = all_indices != (task_id.unsqueeze(0).expand(10, n))
                false = x_max[mask].view(9, n)

                if k == 0:
                    hist_1.append(frag_preds)
                    hist_2.append(frag_preds_aux)
                    hist_3.append([frag_preds_latent])
                elif k > 0:
                    frag_preds = (frag_preds + hist_1[i]*k)/(k+1)
                    frag_preds_aux = (frag_preds_aux + hist_2[i]*k)/(k+1)
                    hist_1[i] = frag_preds
                    hist_2[i] = frag_preds_aux
                    hist_3[i].append(frag_preds_latent)

                batch_size = frag_preds.shape[1]

                ### select across the top 2 of likelihood the head  with the lowest entropy
                # buff -> batch_size  x 2, 0-99 val
                frag_preds_ = frag_preds
                #frag_preds_=torch.softmax(frag_preds / args.temperature, dim=1)###################################################
                #frag_preds_ = frag_preds_[:, args.classes_per_exp * rot * (1 - args.control):args.classes_per_exp * (
                #            rot * (1 - args.control) + 1)]

                buff = frag_preds_.max(dim=-1)[0].argsort(dim=0)
                task_predictions.append(buff[-1])

                # buff_entropy ->  2 x batch_size, entropy values
                indices = torch.arange(batch_size)

                y_hats.append(frag_preds_[buff[-1], indices].argmax(dim=1) + (args.classes_per_exp + args.extra_classes)*buff[-1])
                y_taw.append(frag_preds_[task_id.to(torch.int32), indices].argmax(dim=-1) + ((args.classes_per_exp + args.extra_classes) * task_id.to(args.cuda)).to(torch.int32))

                y_aux.append(frag_preds_latent.argmax(dim=2))

                task_ids.append(task_id)
                ys.append(y)

                A.append(true)
                B.append(false)

            y = torch.cat(ys)
            y_hats = torch.cat(y_hats)
            y_taw = torch.cat(y_taw)
            y_aux = torch.cat(y_aux, dim=1)
            task_ids = torch.cat(task_ids)
            task_predictions = torch.cat(task_predictions)
            A = torch.cat(A)
            B = torch.cat(B, dim=1)

            #to filter out the fake elements added before
            a = y%(args.classes_per_exp + args.extra_classes)
            y = y[a < args.classes_per_exp].cpu()
            y_hats = y_hats[a < args.classes_per_exp].cpu()
            y_taw = y_taw[a < args.classes_per_exp].cpu()
            y_aux = y_aux[:, a < args.classes_per_exp].cpu()
            task_ids = task_ids[a < args.classes_per_exp].cpu()
            task_predictions = task_predictions[a < args.classes_per_exp].cpu()
            A = A[a < args.classes_per_exp].cpu()
            B = B[:, a < args.classes_per_exp].flatten().cpu()

            # Imposta il layout dei sottografi (subplots)
            A_np = A.numpy()
            B_np = B.numpy()

            '''fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 riga, 2 colonne
            axs[0].hist(A_np, bins=30, color='skyblue', edgecolor='black')
            axs[0].set_title("Distribuzione picchi task vera")
            axs[0].set_xlabel("Valore")
            axs[0].set_ylabel("Frequenza")
            axs[0].set_xlim(0, 1)
            axs[1].hist(B_np, bins=30, color='salmon', edgecolor='black')
            axs[1].set_title("Distribuzione picchi task errate")
            axs[1].set_xlabel("Valore")
            axs[1].set_ylabel("Frequenza")
            axs[1].set_xlim(0, 1)
            plt.tight_layout()
            plt.show()'''

            bins = np.linspace(0, 1, 51)
            plt.figure(figsize=(8, 5))
            plt.hist(A_np, bins=bins, color='skyblue', edgecolor='black', alpha=0.6, label='True task', density=True)
            plt.hist(B_np, bins=bins, color='salmon', edgecolor='black', alpha=0.6, label='False tasks', density=True)
            plt.title(f"Class Augm. Rot. x{args.class_augmentation}", fontsize=25)
            plt.xlabel("Max Value", fontsize=20)
            plt.ylabel("Frequency", fontsize=20)
            plt.xlim(0, 1)
            plt.legend(fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
            if seed == 0:
                plt.show()

            '''n_sim = 100_000
            a_samples = np.random.choice(A_np, size=n_sim)
            b_samples = np.random.choice(B_np, size=(n_sim, 9))
            b_max = np.max(b_samples, axis=1)
            if seed == 0:
                prob = np.mean(A_np[:, None] > B_np[None, :])
                prob_1 = np.mean(a_samples > b_max)
            else:
                prop = (prob*seed + np.mean(A_np[:, None] > B_np[None, :]))/(seed+1)
                prob_1 = (prob_1*seed + np.mean(a_samples > b_max)) / (seed + 1)

            print(f"Probabilità che un campione da A sia maggiore di uno da B: {prob:.4f}")

            n_sim = 100_000

            a_samples = np.random.choice(A_np, size=n_sim)
            b_samples = np.random.choice(B_np, size=(n_sim, 9))
            b_max = np.max(b_samples, axis=1)

            print(f"Probabilità che un campione da A sia maggiore di 9 da B: {prob_1:.4f}")'''

            AA.append(A)
            BB.append(B)


            # assign +1 to the confusion matrix for each prediction that matches the label
            for i in range(y.shape[0]):
                confusion_mat[y[i], y_hats[i]] += 1
                confusion_mat_taw[y[i], y_taw[i]] += 1
                for j in range(10):
                    #m = int(y[i]/(args.classes_per_exp * args.class_augmentation))
                    #confusion_mat_aux[y[i]-m*(args.classes_per_exp * (args.class_augmentation - 1)), y_aux[j, i]+(args.classes_per_exp * args.class_augmentation)*j] += 1
                    confusion_mat_aux[y[i], y_aux[j, i] + (args.classes_per_exp * args.class_augmentation) * j] += 1

            # task confusion matrix and forgetting mat
            for j in range(strategy.experience_idx + 1):
                i = strategy.experience_idx
                acc_conf_mat_task = confusion_mat[j * (args.classes_per_exp + args.extra_classes):(j + 1) * (args.classes_per_exp + args.extra_classes),j * (args.classes_per_exp + args.extra_classes):(j + 1) * (args.classes_per_exp + args.extra_classes)].diag().sum() / confusion_mat[i * (args.classes_per_exp + args.extra_classes):(i + 1) * (args.classes_per_exp + args.extra_classes),:].sum()
                strategy.confusion_mat_task[i][j] = acc_conf_mat_task

            accuracy = confusion_mat.diag().sum() / confusion_mat.sum()
            accuracy_taw = confusion_mat_taw.diag().sum() / confusion_mat_taw.sum()
            task_accuracy = (task_predictions == task_ids).sum() / y_hats.shape[0]

            acc.append(accuracy.item())
            taw.append(accuracy_taw.item())
            task.append(task_accuracy.item())
    acc_.append(acc)
    taw_.append(taw)
    task_.append(task)


    ####codice per exp 05/08
    dist_1 = []
    dist_2 = []
    dist_3 = []
    '''for task in range(10):
        original = hist_3[task][0]  # shape: (10, 1000, 10)
        transformed_list = hist_3[task][1:]  # lista di (10, 1000, 10)
        transformed_tensor = torch.stack(transformed_list)  # shape: (n, 10, 1000, 10)
        diff = transformed_tensor - original  # shape: (n, 10, 1000, 10)
        distances = torch.norm(diff, dim=-1)
        mean_distance = distances.mean(dim=0)
        dist.append(mean_distance)'''

    '''for task in range(10):
        for t1, t2 in combinations(hist_3[task], 2):
            diff = t1 - t2  # shape: (10, 1000, 10)
            dist = torch.norm(diff, dim=-1)  # distanza per ogni logit -> (10, 1000)
            dist_1.append(dist)
        dist_2.append(torch.stack(dist_1).mean(dim=0))
    dist = torch.cat(dist_2, dim=1)
    t = task_ids.cpu().numpy()

    print(len(t))
    print(dist.shape)
    conf_mat_t = np.zeros((10, 10))
    for i in range(len(t)):
        for j in range(10):
            conf_mat_t[t[i], j] += dist[j][i]
    plt.imshow(conf_mat_t / conf_mat_t.sum(axis=0), cmap='viridis')
    plt.colorbar()
    plt.show()'''


    print(f"SEED: {seed}")
    tag_mean = np.mean(np.array(acc_).T, axis=1)
    tag_std = np.std(np.array(acc_).T, axis=1)
    taw_mean = np.mean(np.array(taw_).T, axis=1)
    taw_std = np.std(np.array(taw_).T, axis=1)
    task_mean = np.mean(np.array(task_).T, axis=1)
    task_std = np.std(np.array(task_).T, axis=1)

    #confusion_mat_aux = np.zeros((s, s))
    #for i in range(s):
    #    for j in range(s):
    #        if (i % (args.classes_per_exp*args.class_augmentation)) > args.classes_per_exp:
    #            pass
    #        elif i == j:
    #            confusion_mat_aux[i, j] += taw_mean
    #        elif int(i/(args.classes_per_exp*args.class_augmentation)) == int(j/(args.classes_per_exp*args.class_augmentation)):
    #            confusion_mat_aux[i, j] += (1-tag_mean)/(args.classes_per_exp*args.class_augmentation - 1)
    #        else:
    #            confusion_mat_aux[i, j] += 1/(args.classes_per_exp*args.class_augmentation)

    plt.imshow(confusion_mat_aux)
    plt.title("Stacked Confusion Matrix " + args.dataset)
    plt.xlabel("predicetd class")
    plt.ylabel("original class")
    plt.colorbar()
    plt.show()

    #points_of_interest = [0, int(args.n_aug/2), args.n_aug]
    points_of_interest = range(args.n_aug)
    for p in points_of_interest:
        print(f"number of augmentation = {p},     TAG = {tag_mean[p]*100:.2f} ± {tag_std[p]*100:.2f}, TAW = {taw_mean[p]*100:.2f} ± {taw_std[p]*100:.2f},   T = {task_mean[p]*100:.2f} ± {task_std[p]*100:.2f}")

    #print(mean_t)
    #print(std_t)

    #print(mean_f)
    #print(std_f)

    #print(mean_e)
    #print(std_e)

    '''z = np.arange(20)
    plt.figure(figsize=(10, 6))
    plt.plot(z, tag_mean, label='TTDA', color='blue')
    plt.fill_between(z, tag_mean - tag_std, tag_mean + tag_std, color='blue', alpha=0.2)
    #plt.plot(z, taw_mean, label='TAW Mean', color='green')
    #plt.fill_between(z, taw_mean - taw_std, taw_mean + taw_std, color='green', alpha=0.2)
    #plt.plot(z, task_mean, label='TASK Mean', color='red')
    #plt.fill_between(z, task_mean - task_std, task_mean + task_std, color='red', alpha=0.2)
    plt.axhline(y=tag_mean[0], color='black', linestyle='--', label='Baseline')
    plt.xlabel('Augmentation number')
    plt.ylabel('TAG')
    plt.title('Cifar100')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()'''

    #if strategy.experience_idx == 9:
    #    # test_teachers(strategy, strategy.test_scenario[:i+1] )
    #    print('a')
    #    SVCCA_starter(strategy)
    #    print('b')

    #    print("Run Explainability")

    #    #run_explainability_tools(strategy)
    #    print('c')
        #input()

AA = torch.cat(AA)
BB = torch.cat(BB)

print(f"{AA.mean()} ± {AA.var()}")
print(f"{BB.mean()} ± {BB.var()}")