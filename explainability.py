import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from lime import lime_image
from skimage.segmentation import felzenszwalb
from utils.generic import freeze_model
from models.gated_resnet32 import GatedConv2d
from torch.nn.functional import softmax
from parse import args
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
from torch.utils.data import DataLoader
import torch.nn.functional as F
from svcca import cca_core
import math
from matplotlib.ticker import MaxNLocator, MultipleLocator


def _plot_helper(arr, xlabel, ylabel):
    plt.plot(arr, lw=2.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    #plt.show()  # capire ceh succede


# --------------------------------------------
# OCCLUSION EXPLANATION TOOL
# --------------------------------------------
def occlusions_heatmap(model, image, label_idx=None, patch_size=8, stride=4, device='cuda', task_ids=0):
    model.eval()
    image = image.unsqueeze(0).to(device)
    _, _, H, W = image.shape

    lower = task_ids * args.classes_per_exp*args.class_augmentation
    upper = lower+10
    args.class_augmentation

    with torch.no_grad():
        output = model(image)
        print(output.shape)
        input()
        base_pred = output[:, lower:upper]
        print(base_pred.shape)
        # aux_pred = output[:, args.n_classes + task_ids].unsqueeze(1)
        # pred = torch.cat([base_pred, aux_pred], dim=1)
        probs = softmax(base_pred, dim=1)

    if label_idx is not None and lower <= label_idx < upper:
        label_idx = label_idx % args.classes_per_exp
    else:
        label_idx = base_pred.argmax(dim=1).item()

    base_conf = base_pred[0, label_idx].item()
    avg_confidences = [probs[0, label_idx].item()]
    num_rows = len(range(0, H - patch_size + 1, stride))
    num_cols = len(range(0, W - patch_size + 1, stride))
    heatmap = np.zeros((num_rows, num_cols))

    for i, y in enumerate(range(0, H - patch_size + 1, stride)):
        for j, x in enumerate(range(0, W - patch_size + 1, stride)):
            occluded = image.clone()
            occluded[:, :, y:y + patch_size, x:x + patch_size] = 0.5

            with torch.no_grad():
                out = model(occluded)
                occl_pred = out[:, lower:upper]
                # aux_occl = out[:, args.n_classes + task_ids].unsqueeze(1)
                # combined = torch.cat([occl_pred, aux_occl], dim=1)
                probs_occl = softmax(occl_pred, dim=1)
                conf = occl_pred[0, label_idx].item()
                avg_confidences.append(probs_occl[0, label_idx].item())
                heatmap[i, j] = base_conf - conf

    mean_conf = sum(avg_confidences) / len(avg_confidences)

    return heatmap, mean_conf, label_idx + args.classes_per_exp * task_ids


# --------------------------------------------
# LIME EXPLANATION TOOL
# --------------------------------------------
def lime_tool(input_image, model, label=0, task_ids=0):
    model.eval()

    def segment_fn(img):
        return felzenszwalb(img, scale=50, sigma=0.5, min_size=10)

    def predict_fn(images):
        images = torch.tensor(np.transpose(images, (0, 3, 1, 2)), dtype=torch.float32).to(args.device)
        with torch.no_grad():
            logits = model(images)
            sub_pred = logits[:, task_ids * args.classes_per_exp:(task_ids + 1) * args.classes_per_exp]
            # aux = logits[:, args.n_classes + task_ids].unsqueeze(1)
            probs = softmax(sub_pred, dim=1)
        return probs.cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image=input_image.permute(1, 2, 0).cpu().numpy(),
        classifier_fn=predict_fn,
        top_labels=5,
        hide_color=0,
        segmentation_fn=segment_fn,
        num_samples=2000
    )
    return explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                          hide_rest=False)


# Normalize and prepare image for plotting (PyTorch version)
def normalize_image(img_tensor):
    # Center tensor and normalize to std of 0.1 (as in deprocess_image)
    img_tensor = img_tensor - img_tensor.mean()
    img_tensor = img_tensor / (img_tensor.std() + 1e-5)
    img_tensor = img_tensor * 0.1

    # Shift to [0, 1] range
    img_tensor = img_tensor + 0.5
    img_tensor = img_tensor.clamp(0, 1)

    # Convert from (C, H, W) to (H, W, C) and to numpy
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

    # Scale to [0, 255] and convert to uint8
    img_np = (img_np * 255).astype('uint8')

    return img_np


# --------------------------------------------
# MAXIMAL ACTIVATION TOOL
# --------------------------------------------
def maximal_response(model, channel=0, number_conv=2):
    model.eval()

    conv_number = 1
    for m in model.modules():
        if isinstance(m, GatedConv2d):
            if conv_number == number_conv:
                target_layer = m
                break
            conv_number = conv_number + 1

    if not target_layer:
        raise ValueError("No suitable GatedConv2d layer found.")

    activation = {}
    hook = target_layer.register_forward_hook(lambda m, inp, out: activation.update({"value": out[:, channel, :, :]}))

    shape = (1, 3, 64, 64) if args.dataset == "TinyImageNet" else (1, 3, 32, 32)  # create random  image

    input_image = torch.randn(*shape, requires_grad=True, device=args.device)
    optimizer = torch.optim.Adam([input_image], lr=0.1)

    for _ in range(1200):
        optimizer.zero_grad()
        model(input_image)
        act = activation.get("value")  # get the activation value
        if act is None:
            raise RuntimeError("Activation hook not triggered.")
        loss = -act.mean()
        loss.backward()

        grads = input_image.grad
        grads_normalized = grads / (torch.sqrt(torch.mean(grads ** 2)) + 1e-5)
        input_image.grad = grads_normalized
        optimizer.step()

    hook.remove()
    return input_image.detach()


def visualize_occlusion_and_lime(image, heatmaps, label_original, titles=None, cmap='jet', save_path=None,
                                 figsize=(25, 8), start_index=0):
    """
    Visualize occlusion heatmaps from multiple subnetworks for a single input image.

    Args:
        image (Tensor): Original image [3, H, W] (PyTorch tensor)
        heatmaps (List[Tuple[np.array, float]]): List of tuples (heatmap, average_probability)
        titles (List[str]): Titles for each heatmap
        cmap (str): Colormap for heatmaps
        save_path (str): If set, saves the plot to this file
        figsize (tuple): Size of the full figure
    """

    image = image.cpu().permute(1, 2, 0).numpy()
    image = (image - image.min()) / (image.max() - image.min())  # Normalize image
    H, W, _ = image.shape

    ncols = int(len(heatmaps) / 2 + 1)
    nrows = 2  # +1 for input image

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    # Show input image in the first subplot
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[0].set_title(f"label Image {label_original} ")

    # Show occlusion heatmaps
    for i, (heatmap, avg_prob) in enumerate(heatmaps):

        ax = axes[i + 1]  # +1 because 0 is for the original image
        norm_heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)

        # Resize heatmap
        heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).float()
        heatmap_resized = F.interpolate(heatmap_tensor, size=(H, W), mode='bilinear', align_corners=False)
        heatmap_resized = heatmap_resized.squeeze().cpu().numpy()
        ax.imshow(image)
        ax.imshow(heatmap_resized, cmap=cmap, alpha=0.5)
        ax.axis('off')
        if titles:
            ax.set_title(titles[i])
        else:
            ax.set_title(f"SubNet {i + start_index} - {avg_prob:.2f}")

    for j in range(len(heatmaps) + 1, nrows * ncols):
        axes[j].axis('off')

    plt.tight_layout()

    if save_path:
        save_path = f"{save_path}_occlusion.png"
        plt.savefig(save_path)
    else:
        #plt.show()
        pass


def get_one_sample_per_class(dataloader, class_samples):
    for i, (images, labels, task_id) in enumerate(dataloader):
        for img, label in zip(images, labels):
            label = label.item()  # convert tensor to int
            if label not in class_samples:
                class_samples[label] = img

    return class_samples


# --------------------------------------------
# EXPLAINABILITY RUNNER
# --------------------------------------------
def run_explainability_tools(strategy):

    class_samples = {}
    for i in range(strategy.test_scenario.nb_tasks):
        test_task = strategy.test_scenario[i]
        dataloader = DataLoader(test_task, batch_size=20, shuffle=False, num_workers=8)
        class_samples = get_one_sample_per_class(dataloader, class_samples)

    test_set = list(class_samples.values())
    labels = list(class_samples.keys())
    root_folder = f"logs/{args.dataset}/explainability/"
    os.makedirs(root_folder, exist_ok=True)
    maximization = True
    # set zero to all the output neurons that are not in this experience, ma questa operazione viene fatta per il nuovo modello
    # strategy.fresh_model.set_output_mask(i, train_taskset.get_classes())

    for image_number, image in enumerate(test_set):

        label = labels[image_number]
        heatmaps = []
        titles_occlusion = []
        # lime_images=[]
        # lime_masks=[]
        if image_number != 0:
            maximization = False
        else:
            False

        for idx in range(args.n_experiences):

            print(f"Subnetwork used: {idx} ")

            # teacher_model = deepcopy(strategy.teacher_models[idx])
            # teacher_model.set_output_mask(idx, strategy.test_scenario[i].get_classes())

            model = freeze_model(deepcopy(strategy.model))
            strategy.pruner.set_gating_masks(model, idx, weight_sharing=args.weight_sharing, distillation=True)

            model.load_bn_params(idx)
            model.exp_idx = idx

            max_imgs = []
            # max_imgs_teacher=[]
            convs = [1, 6, 11, 16]

            if maximization:

                for conv in convs:

                    for i in range(16):
                        max_imgs.extend(maximal_response(model, channel=i, number_conv=conv))
                        #   max_imgs_teacher.extend(maximal_response(teacher_model, channel=i,number_conv=conv))

                    # unify the vectors
                    # max_imgs=max_img#s+max_imgs_teacher

                    # Plot 4 rows × 8 columns
                    fig, axes = plt.subplots(2, 8, figsize=(16, 8))
                    for i, ax in enumerate(axes.flat):
                        img = normalize_image(max_imgs[i])
                        ax.imshow(img)
                        ax.axis('off')
                        model_idx = "student" if i < 16 else "teacher"
                        ax.set_title(f"{model_idx} - Ch {i % 16}", fontsize=8)

                    plt.tight_layout()
                    os.makedirs(f"{root_folder}/layer_visualization", exist_ok=True)
                    plt.savefig(f"{root_folder}/layer_visualization/max_response_grid_netowrk_{idx}_conv{conv}.png",
                                dpi=300, bbox_inches='tight')

            # lime_img, lime_mask = lime_tool(image, model, task_ids=idx)
            # lime_masks.append(lime_mask)
            # lime_images.append(lime_img)

            heatmap, mean_val, label_idx = occlusions_heatmap(model, image, label, task_ids=idx)
            title_occlusion = f"Subnetwork {idx}\nprediction {label_idx} {mean_val}"
            titles_occlusion.append(title_occlusion)
            heatmaps.append((heatmap, mean_val))

        stringa = f"{root_folder}image_label_{label}"

        visualize_occlusion_and_lime(image, heatmaps, label, save_path=stringa, titles=titles_occlusion)


def find_conv(model, number_conv):
    conv_number = 1
    target_layer = None
    for m in model.modules():
        if isinstance(m, GatedConv2d):
            if conv_number == number_conv:
                target_layer = m
                break
            conv_number = conv_number + 1

    if target_layer == None:
        print(f"convolution number {number_conv} not found in model\n")
        print(model)
        input("Sanity check")

    return target_layer


def SVCCA_starter(strategy, skip_conv=1, plot=True):
    '''
     Input
     strategy is the main Class that wraps also the network
     test_set, imported test set
     list of index of the convolution that needs to be used
    '''

    print('inizia')

    test_set = strategy.test_scenario[:args.n_experiences]
    dataloader = DataLoader(test_set, batch_size=10000, shuffle=False, num_workers=8)

    total_convolutions = 32  # fare il conto
    activations_subnets = []
    for i, (x, y, task_id) in enumerate(dataloader):
        print('michele')

        # create a temporary model copy
        model = freeze_model(deepcopy(strategy.model))
        x = x.to(args.device)
        for j in range(args.n_experiences):

            strategy.pruner.set_gating_masks(model, j, weight_sharing=args.weight_sharing, distillation=True)
            model.load_bn_params(j)
            model.exp_idx = j
            hooks = []
            activations = {}
            for number_convolution in range(1, total_convolutions, skip_conv):
                target_layer = find_conv(model, number_convolution)

                def make_hook(layer_id):
                    return lambda m, inp, out: activations.update({f"conv_{layer_id}": out})

                hook = target_layer.register_forward_hook(make_hook(number_convolution))

                hooks.append(hook)

            model(x)
            activations = {k: v.detach().cpu() for k, v in activations.items()}

            activations_subnets.append(activations)
            # Remove all hooks
            for hook in hooks:
                hook.remove()

        break  # ensure just one cicle

    os.makedirs(f"svcca_explainability_{args.class_augmentation}", exist_ok=True)
    plt.figure(figsize=(12, 6))

    for subnet_index in range(len(activations_subnets) - 1):
        print('processing')

        # apply svcca using the activations of subnet_index  and subnet_index+1
        # perform svcca for each convolution
        activation_one = activations_subnets[subnet_index]
        activation_two = activations_subnets[subnet_index + 1]
        number_convolutions_line = len(activation_one.keys()) / 3
        plot_name = f"Comparison_{subnet_index}_{subnet_index + 1}"

        fig, axes = plt.subplots(2,2, figsize=(15, 8))
        axes = axes.flatten()  # Flatten in case of 2D axes array

        # perform the analisys for each convolutional layer saved
        cca_comparisons = []
        for idx, (key, act_one) in enumerate(activation_one.items()):
            act_two = activation_two[key]
            act_one = act_one.to("cpu")
            act_two = act_two.to("cpu")

            # permute the input
            act_one = act_one.permute(0, 2, 3, 1)
            act_two = act_two.permute(0, 2, 3, 1)

            # take the dimentionality and reshape
            num_datapoints, h, w, channels = act_one.shape
            num_datapoints_two, h_two, w_two, channels_two = act_two.shape

            assert h == h_two or w == w_two or channels == channels_two, f"the two convolutions shold have the same dimensionality"

            f_acts_one = act_one.reshape((num_datapoints * h * w, channels))
            f_acts_two = act_two.reshape((num_datapoints * h * w, channels))

            # perform svcca
            f_results = cca_core.get_cca_similarity(f_acts_one.numpy().T, f_acts_two.numpy().T, epsilon=1e-10,
                                                    verbose=False)
            # print(f_results.keys())

            cca_comparisons.append(np.sum(f_results["cca_coef1"]) / len(f_results["cca_coef1"]))

            if idx%10==0:

                ax = axes[int(idx/10)]
                ax.plot(f_results["cca_coef1"], lw=2.0)
                ax.set_title(f"Layer {key}")
                ax.set_xlabel("CCA Coef idx")
                ax.set_ylabel("CCA coef value")
                ax.grid(True)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MultipleLocator(0.2))

                fig.suptitle(plot_name, fontsize=8, y=1.02)

            # Hide unused axes
            #  for j in range(idx + 1, len(axes)):
            #    axes[j].axis('off')
        plt.subplots_adjust(hspace=2.0)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # top=0.95 leaves space at the top

        plt.tight_layout()
        #plt.show()
        fig.savefig(f"svcca_explainability_{args.class_augmentation}/{plot_name}.png",  dpi=300, bbox_inches="tight")

        x = list(activation_one.keys())

        y = cca_comparisons

        plt.plot(x, y, marker='o', linestyle='-', label=plot_name)  # Line with dots at each point

#    plt.xlabel('Conv numbers')
 #   plt.ylabel('Normalized SVCCA')
  #  plt.title('SVCCA progression into the network')
   # plt.xticks(rotation=45)  # Rotate x labels if needed
    #plt.legend(title="Subnet Pairs", loc='upper right', fontsize='small')
    #plt.tight_layout()
    #plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Show the plot

    #plt.savefig(f"svcca_explainability/Compressed.png", dpi=300, bbox_inches='tight')
    #plt.show()

    return