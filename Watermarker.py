import os
import math
import shutil
import glob
import json
import torch
from tabulate import tabulate
from time import time
from torch.nn.functional import cross_entropy
from utils import top_n_accuracy, lorem_ipsum, tobits, weights_summary, create_watermark_configuration
import numpy as np
from resnet import ResNet18
from densenet import DenseNet121, DenseNet169
from xception import Xception
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision.datasets import GTSRB
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import prune
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
import matplotlib
from scipy.io import savemat
from scipy.special import rel_entr

sns.set_style("white")

class Watermarker:

    def __init__(self, embed, device, task, key=123456, net='resnet18', dataset='cifar10'):
        self._embed = embed
        self._task = task
        self._key = key
        self._device = device
        self._device_count = torch.cuda.device_count()
        self._net = net.lower()
        self._dataset = dataset
        self._config_json = None
        self._spread = None
        self._C = None
        self._bits = None
        self._message = None
        self._watermark_host_layers = []
        self._watermark_chunks = []
        self._watermark_strength = []
        self._watermark_host_coords = []
        self._out_dir = ''
        self._model = None
        self._model_input_shape = None
        self._num_classes = None

        self._dataloader_train = None
        self._dataloader_valid = None
        self._dataloader_test = None
        self._batch = 64

        return

    def host_layer_exists(self):
        """
        Check if target layers exists inside the network model
        """
        found = 0
        for i, (name, param) in enumerate(self._model.named_parameters()):
            if name in self._watermark_host_layers:
                found += 1
        return found > 0

    def host_layers_lengths(self):
        """
        Return the number of weights of the host layers
        """
        lengths = []
        for i, (name, param) in enumerate(self._model.named_parameters()):
            if name in self._watermark_host_layers:
                lengths.append(param.numel())
        return lengths

    def ExperimentID(self, args, root='models', same_dataset=False):
        """
        Assign a unique output directory to the experiment and avoid to overwrite the same setups
        """

        # No directory needed when planning
        if self._task == 'plan':
            return

        # Include the mode in the directory name
        if self._task == 'train':
            exp_id = f"{self._dataset}_{self._net}_baseline"
        elif self._task == 'embed':
            exp_id = f"{self._dataset}_{self._net}_B{len(self._bits)}_S{self._spread}_C{self._C}"
        else:
            exp_id = f"{self._dataset}_{self._net}_{self._task}_B{len(self._bits)}_S{self._spread}_C{self._C}"

        # Avoid overwriting multiple runs of the same setup by creating a new progressive subfolder
        idx = 0
        if os.path.exists(os.path.join(root, exp_id)):
            subdirs = glob.glob(f"{root}/{exp_id}/*/")
            if len(subdirs) > 0:
                for sd in subdirs:
                    if sd.endswith('/'):
                        sd = sd[:-1]
                    postfix = os.path.basename(sd)
                    if postfix.isnumeric() and int(postfix) > idx:
                        idx = int(postfix)

        self._out_dir = os.path.join(root, exp_id, str(idx+1))

        # When re-training on the same dataset used for embedding, change 'finetune' to 'incremental' in path
        if same_dataset:
            self._out_dir = self._out_dir.replace('finetune', 'incremental')

        os.makedirs(self._out_dir, exist_ok=True)

        # Dump command line arguments in the output dir
        with open(os.path.join(self._out_dir, "args.txt"), 'w') as f:
            f.write(str(args))

        return

    def CreateModel(self, input_shape, from_model=None, summary=True):

        assert self._net in ['densenet121', 'densenet169', 'resnet18', 'xception'], f"Architecture {self._net} not supported."

        # For all the modes except baseline training, load standard deviations (used in the embedding formulas) for the
        # baseline target layers from a .json configuration file that is generated during the initial baseline training
        if self._task != 'train':
            assert os.path.exists(f"config/config-{self._net}-cifar10.json"), f"Configuration not found."
            with open(f"config/config-{self._net}-cifar10.json") as f:
               self._config_json = json.load(f)

        self._model_input_shape = input_shape

        self._num_classes = 10
        if self._dataset == "cifar10" or (from_model is not None and "cifar10" in from_model):
            self._num_classes = 10
        elif self._dataset == "gtsrb" or (from_model is not None and "gtsrb" in from_model):
            self._num_classes = 43

        if self._net == 'resnet18':
            self._model = ResNet18(num_classes=self._num_classes, norm_type='bn')
        elif self._net == 'densenet121':
            self._model = DenseNet121(num_classes=self._num_classes)
        elif self._net == 'densenet169':
            self._model = DenseNet169(num_classes=self._num_classes)
        elif self._net == 'xception':
            self._model = Xception(num_classes=self._num_classes)

        if from_model is not None:
            sd = torch.load(from_model)
            model_dict = self._model.state_dict()
            pretrained_dict = {k: v for k, v in sd.items() if k.find('classifier') == -1}
            model_dict.update(pretrained_dict)
            sd = model_dict
            self._model.load_state_dict(sd["model_state_dict"])

            if self._dataset == "gtsrb":
                if hasattr(self._model, 'linear'):
                    self._model.linear = torch.nn.Linear(self._model.linear.in_features, 43)
                if hasattr(self._model, 'fc'):  # XCeption
                    self._model.fc = torch.nn.Linear(self._model.fc.in_features, 43)
                else:  # Densenet
                    self._model.classifier = torch.nn.Linear(self._model.classifier.in_features, 43)

        if self._dataset == "cifar100":
            if hasattr(self._model, 'classifier'):
                self._model.classifier = torch.nn.Linear(self._model.classifier.in_features, 100)
            if hasattr(self._model, 'fc'):      # XCeption
                self._model.fc = torch.nn.Linear(self._model.fc.in_features, 100)
            else:    # Densenet
                self._model.linear = torch.nn.Linear(self._model.linear.in_features, 100)

        self._model.to(self._device)

        if summary:
            print(weights_summary(self._model))
        return

    def CreateWatermark(self, n_bits, spread, C, host_layers=None, host_shares=None):

        if self._task in ['train']:
            return

        # this is the case when we plan just to check layer names and not their occupancy
        if self._task in ['plan'] and len(host_layers) == 1 and host_layers[0] == "":
            self._watermark_host_layers = []
            return

        msg = lorem_ipsum(n_bits)
        binary_seq = tobits(msg)

        self._bits = binary_seq
        self._message = np.repeat(2 * np.array(binary_seq) - 1, spread)
        self._spread = spread
        self._watermark_host_layers = host_layers
        self._C = C

        # Check if the host layers exist
        assert self.host_layer_exists(), "One or more host layers not found."
        assert len(binary_seq) == n_bits, "The lorem ipsum watermarking string is too short for this payload"

        # If no host shares are provided, split equally the watermark among all the host layers
        if host_shares is None:
            wat_chunk_length = [len(x) for x in np.array_split(np.arange(len(self._message)), len(self._watermark_host_layers))]
        # Otherwise, use the shares to assign the desired watermark chunk to each host layer
        else:
            wat_chunk_length = [math.floor(x * len(self._message)) for x in host_shares]
            wat_chunk_length[-1] = len(self._message) - sum(wat_chunk_length[:-1])

        interval = np.cumsum([0, ] + wat_chunk_length)

        self._watermark_chunks = []
        for i in range(0, len(interval)-1):
            self._watermark_chunks.append(self._message[interval[i]:interval[i+1]])

        def laplacian_distribution(mu, sigma, samples):
            if self._key is not None:
                np.random.seed(self._key)
            return np.random.laplace(loc=mu, scale=sigma, size=samples)

        variance_info = []
        for i, host_layer in enumerate(host_layers):

            stdev = float(self._config_json[host_layer]["std"])
            ksigma = self._C * stdev * (1 / np.sqrt(2))   # so that var(laplacian) = var(weights)
            laplace = laplacian_distribution(mu=0, samples=wat_chunk_length[i], sigma=ksigma)

            variance_info.append(f"Std.dev. for baseline '{host_layer.upper()}' is {stdev:.5f}")
            variance_info.append(f"Variance for baseline '{host_layer.upper()}' is {stdev**2:.5f}")
            variance_info.append(f"Std.dev. for watermark values is {np.sqrt(np.var(laplace)):.5f}")
            variance_info.append(f"Variance for watermark values is {np.var(laplace):.5f}")

            self._watermark_strength.append(np.abs(laplace))

        self._watermark_host_coords = []
        k = 0
        torch.manual_seed(self._key)
        for i, (name, param) in enumerate(self._model.named_parameters()):
            if name in self._watermark_host_layers:
                n_weights = len(param.view(-1))
                indices = torch.randperm(n_weights)[:wat_chunk_length[k]]
                self._watermark_host_coords.append(indices.numpy())
                k += 1

        return variance_info

    def CreateDataset(self, batch_size=128, same_dataset=False):

        self._batch = batch_size

        if self._dataset in ["cifar10", "cifar100"]:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        elif self._dataset in ["gtsrb"]:
            transform_train = transforms.Compose([
                transforms.Resize(40),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation(40),
                transforms.ColorJitter(brightness=.5, hue=.3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.3431, 0.3130, 0.3224), (0.1991, 0.1955, 0.2046))
            ])

            transform_test = transforms.Compose([
                transforms.Resize(40),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.3431, 0.3130, 0.3224), (0.1991, 0.1955, 0.2046))
            ])

        if self._dataset == "cifar10":
            DATASET = CIFAR10
        elif self._dataset == "cifar100":
            DATASET = CIFAR100
        else:
            DATASET = GTSRB

        if self._dataset in ["cifar10", "cifar100"]:
            train_dataset = DATASET('datasets', train=True, transform=transform_train, download=True)
            valid_dataset = DATASET('datasets', train=True, transform=transform_test, download=True)
            test_dataset = DATASET('datasets', train=False, transform=transform_test, download=True)
        else:
            train_dataset = DATASET('datasets', split='train', transform=transform_train, download=True)
            valid_dataset = DATASET('datasets', split='train', transform=transform_test, download=True)
            test_dataset = DATASET('datasets', split='test', transform=transform_test, download=True)

        # Training on a subset of 70% of the original data (for finetuning on the same task only)
        if same_dataset:

            subset_train_size = int(0.7 * len(train_dataset))
            indices = torch.randperm(len(train_dataset)).tolist()

            train_dataset = Subset(train_dataset, indices[:subset_train_size])
            valid_dataset = Subset(valid_dataset, indices[:subset_train_size])

        self._dataloader_train = DataLoader(train_dataset, batch_size=self._batch, shuffle=True, num_workers=4, pin_memory=True)
        self._dataloader_valid = DataLoader(valid_dataset, batch_size=100, shuffle=True, num_workers=4, pin_memory=True)
        self._dataloader_test = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
        return

    def Train(self, lr=0.1, epochs=200, save_best_only=False, out_log='log.csv'):

        optimizer = torch.optim.SGD(self._model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

        if self._dataset in ["cifar10", "cifar100"]:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 150], 0.1)
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.65 ** epoch)

        with open(os.path.join(self._out_dir, out_log), 'a') as f:
            f.write("Epoch;Train_loss;Val_loss;Val_acc_top1;Val_acc_top3;BER;Time\n")

        best_valid_loss = float('inf')

        for ep in range(epochs):

            self._model.train()
            tot_loss = 0

            start = time()

            for i, (data, target) in tqdm(enumerate(self._dataloader_train), total=len(self._dataloader_train)):
                data = data.to(self._device)
                target = target.to(self._device)

                optimizer.zero_grad()

                predictions = self._model(data)
                loss = cross_entropy(predictions, target)
                loss.backward()
                optimizer.step()

                # Watermark here (this is True only for 'Embed' mode in command line arguments)
                if self._embed:
                    self.EmbedWatermark()

                tot_loss += loss.item()

            # On each epoch's end
            scheduler.step()

            ber = 0.0
            if self._task in ['embed', 'finetune']:
                _, ber = self.ReadWatermark()

            train_loss = tot_loss / len(self._dataloader_train)
            ep_time = time() - start

            val_loss, val_acc = self.Validate(False)

            # Save only best model using validation loss (can do this also with validation accuracy)
            if save_best_only:
                if val_loss < best_valid_loss:
                    best_valid_loss = val_loss
                    print(f"\nBest validation loss: {best_valid_loss}. Saving best model for epoch: {ep + 1}\n")
                    torch.save({'epoch': ep+1,
                                'loss': val_loss,
                                'acc': val_acc,
                                'model_state_dict': self._model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                }, os.path.join(self._out_dir, 'best_model.pt'))
            else:
                torch.save(self._model, os.path.join(self._out_dir, f'model_{ep+1}.pt'))

            # Append each epoch metrics to the .csv log file of the current experiment
            with open(os.path.join(self._out_dir, out_log), 'a+') as f:
                f.write(f"{ep+1};{train_loss:.3f};{val_loss:.5f};{val_acc[0]:.5f};{val_acc[1]:.5f};{ber};{ep_time:.5f}\n")
                f.flush()

        if self._task == 'train':
            create_watermark_configuration(self._model, f"config/config-{self._net}-{self._dataset}.json")

        return

    def Validate(self, is_test=False, topk=5, to_log=True):

        dataloader = self._dataloader_test if is_test else self._dataloader_train

        self._model.eval()

        acc_top1, acc_topK = 0, 0
        tot_loss = 0
        count = 0
        with torch.no_grad():
            for (data, labels) in dataloader:

                data, labels = data.to(self._device), labels.to(self._device)

                predictions = self._model(data)
                tot_loss += cross_entropy(predictions, labels).item()

                pred = predictions.max(1, keepdim=True)[1]
                acc_top1 += pred.eq(labels.view_as(pred)).sum().item()

                acc_topK += top_n_accuracy(predictions, labels, n=5)

                count += data.size(0)

        acc_top1 = acc_top1 / count
        acc_topK = acc_topK / count
        tot_loss = tot_loss / count

        if is_test and to_log:
            with open(os.path.join(self._out_dir, "results.txt"), 'a+') as f:
                f.write(f"\nACC TOP 1: {acc_top1:.3f}\n")
                f.write(f"ACC TOP {topk}: {acc_topK:.3f}\n")
                f.write(f"TER TOP 1: {(1-acc_top1):.3f}\n")
                f.write(f"TER TOP {topk}: {(1-acc_topK):.3f}")

        return tot_loss, (acc_top1, acc_topK)

    def EmbedWatermark(self):
        """
        Embed the watermark into network weights
        """
        k = 0
        for i, (name, param) in enumerate(self._model.named_parameters()):
            if name in self._watermark_host_layers:
                weights = param.view(-1)

                # mMust turn off gradients to fiddle with weights mid-training
                with torch.no_grad():
                    weights[self._watermark_host_coords[k]] = \
                        torch.tensor(self._watermark_chunks[k] * self._watermark_strength[k], device=self._device).type(
                            torch.float)
                weights = torch.reshape(weights, param.shape)
                param.data = weights
                k += 1
        return

    def ReadWatermark(self, save_txt=False):
        """
        Recover the watermark
        """
        watermark = []
        k = 0
        for i, (name, param) in enumerate(self._model.named_parameters()):
            if name in self._watermark_host_layers:
                weights = param.view(-1)
                watermark.append(weights[self._watermark_host_coords[k]])
                k += 1

        # Reshape according to the spread in a [n_bits x spread] matrix
        w_shape = (int(len(self._message) / self._spread), self._spread)
        watermark = torch.hstack(watermark).reshape(w_shape).sum(dim=1)

        rec_message = (watermark > 0).type(torch.int)

        bits_ok = rec_message.eq(torch.Tensor(self._bits).to(self._device)).sum().item()
        ber = (len(self._bits) - bits_ok) / len(self._bits)

        if save_txt:
            with open(os.path.join(self._out_dir, "results.txt"), 'a+') as f:
                f.write(f"\nBER: {ber:.4f}")

        return rec_message.cpu().detach().numpy(), ber

    def PlotDistributions(self, save_png=True, save_data=False, plot_all_weights=False):
        """
        Plot weights distributions for each watermark host layer (watermarked/non watermarked weights) and
        for the whole model
        """
        kwargs = dict(kde_kws={'alpha': 1}, line_kws={'linewidth': 2})

        k = 0
        all_weights = []
        for i, (name, param) in enumerate(self._model.named_parameters()):

            weights = param.view(-1).cpu().detach().numpy()
            if plot_all_weights:
                all_weights.append(weights)

            if name in self._watermark_host_layers:

                if self._task == 'train':
                    not_wat_weights = weights
                else:
                    host_coords = self._watermark_host_coords[k]

                    mask = np.ones(len(weights), dtype=bool)
                    mask[host_coords] = False

                    wat_weights = weights[host_coords]
                    not_wat_weights = weights[mask]

                if save_png:

                    fig_name = f"weights_distributions_{self._dataset}_{self._net}_B{len(self._bits)}"\
                               f"_S{self._spread}_C{self._C}_L{len(self._watermark_host_layers)}_"\
                               f"{name.replace('.conv.weight','')}"

                    plt.figure(figsize=(15, 15), dpi=1200)
                    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
                    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
                    plt.gca().yaxis.offsetText.set_fontsize(24)

                    sns.histplot(not_wat_weights, bins=500, color="#135CA3", element='step',
                                 label=f"Non-watermarked '{name.replace('.conv.weight','')}'", **kwargs)
                    if self._task != 'train':
                        sns.histplot(wat_weights, bins=500, element='step', color="red", #"orange",
                                     label=f"Watermarked '{name.replace('.conv.weight','')}'", **kwargs)

                    plt.ylabel("Weights count", fontsize=30)
                    plt.xlabel("w", fontsize=30)
                    plt.legend(loc=2, fontsize=20)
                    plt.yticks(fontsize=30)
                    plt.xticks(fontsize=30)
                    plt.xlim([-0.05, 0.05])
                    plt.grid(linestyle='--')

                    plt.savefig(os.path.join(self._out_dir, f"{fig_name}.png"), dpi=600)
                    plt.savefig(os.path.join(self._out_dir, f"{fig_name}.pdf"), dpi=1200, format="pdf")
                    plt.close()

                if save_data:

                    data_file = f"weights_data_{self._dataset}_{self._net}_B{len(self._bits)}" \
                                f"_S{self._spread}_C{self._C}_L{len(self._watermark_host_layers)}_" \
                                f"{name.replace('.conv.weight', '')}"

                    # Matlab
                    savemat(os.path.join(self._out_dir, f"{data_file}.mat"),
                             {"nonwatweights": not_wat_weights,
                             "watweights": wat_weights})

                    # Numpy
                    # np.savez(os.path.join(self._out_dir, f"{data_file}.mat", not_wat_weights, wat_weights))
                k += 1

        if plot_all_weights and save_png:

            if self._task == 'train':
                fig_name = f"all_weights_distributions_{self._dataset}_{self._net}"
            else:
                fig_name = f"all_weights_distributions_{self._dataset}_{self._net}_B{len(self._bits)}" \
                           f"_S{self._spread}_C{self._C}_L{len(self._watermark_host_layers)}"

            plt.figure(figsize=(15, 15), dpi=1200)
            plt.gca().yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
            plt.gca().yaxis.offsetText.set_fontsize(24)

            sns.histplot(np.concatenate(all_weights), color="#135CA3", element='step',
                         label="Non-watermarked net" if self._task == 'train' else "Watermarked net", **kwargs)
            plt.ylabel("Weights count", fontsize=30)
            plt.xlabel("w", fontsize=30)
            plt.legend(loc=2, fontsize=30)
            plt.yticks(fontsize=30)
            plt.xticks(fontsize=30)
            plt.xlim([-0.06, 0.06])
            plt.grid(linestyle='--')

            plt.savefig(os.path.join(self._out_dir, f"{fig_name}.png"), dpi=600)
            plt.savefig(os.path.join(self._out_dir, f"{fig_name}.pdf"), dpi=1200, format="pdf")
            plt.close()

        if save_data:
            data_file = f"weights_data_{self._dataset}_{self._net}_B{len(self._bits)}" \
                        f"_S{self._spread}_C{self._C}_L{len(self._watermark_host_layers)}_all_network"

            # Matlab
            savemat(os.path.join(self._out_dir, f"{data_file}.mat"), {"allnetweights": np.concatenate(all_weights)})
        return

    def Plan(self, show_target_only=True):

        if self._task == 'train':
            return

        layer_names, layer_shapes, layer_numels, layer_variance, layer_stddev, is_host, shares, indices = [], [], [], [], [], [], [], []
        k = 0
        for i, (name, param) in enumerate(self._model.named_parameters()):
            if 'weight' in name:
                weights = param.view(-1).cpu().detach().numpy()

                layer_names.append(name)
                layer_shapes.append(str(param.size()))
                layer_numels.append(param.numel())
                layer_variance.append(np.var(weights))
                layer_stddev.append(np.sqrt(np.var(weights)))
                indices.append(i)

                if name in self._watermark_host_layers:
                    is_host.append('*')
                    occupation_p = len(self._watermark_chunks[k]) / len(weights)
                    # Watermark (chunk) does not fit into the layer
                    if occupation_p > 1:
                        shares.append(f'n.a. ({len(self._watermark_chunks[k])})')
                    # It fits
                    else:
                        shares.append(f"{100 * occupation_p:.2f}% ({len(self._watermark_chunks[k])})")
                    k += 1
                else:
                    is_host.append('')
                    shares.append('')

                if show_target_only:
                    if name not in self._watermark_host_layers:
                        layer_names.pop()
                        layer_shapes.pop()
                        layer_numels.pop()
                        layer_variance.pop()
                        layer_stddev.pop()
                        is_host.pop()
                        shares.pop()
                        indices.pop()

        # Gather data for output
        if len(self._watermark_host_layers):
            spreads = [self._spread] * len(layer_names)
            message_lengths = [len(self._bits)] * len(layer_names)
            wat_length = [len(self._message)] * len(layer_names)
        else:
            spreads = ['nd'] * len(layer_names)
            message_lengths = ['nd'] * len(layer_names)
            wat_length = ['nd'] * len(layer_names)

        headers = ["L_index", "L_Name", "L_shape", "L_count", "L_variance", "L_std_dev", "W_spread", "W_mgs",
                   "W_wat", "W_host", "W_share"]

        table = zip(indices, layer_names, layer_shapes, layer_numels, layer_variance, layer_stddev,
                    spreads, message_lengths, wat_length, is_host, shares)

        # Print data
        print(f"{self._net.upper()}")
        print(tabulate(table, headers=headers))

        # Dump data to CSV
        with open(os.path.join(self._out_dir, 'plan.csv'), 'w+', encoding="utf-8") as f:
            f.write(f"{';'.join(headers)}\n")
            for i in range(len(layer_names)):
                f.write(f"{indices[i]};{layer_names[i]};{layer_shapes[i]};"
                        f"{layer_numels[i]};{layer_variance[i]:.5f};{layer_stddev[i]:.5f};"
                        f"{spreads[i]};{message_lengths[i]};{wat_length[i]};{is_host[i]};{shares[i]}\n")

        if self._task == 'plan':
            shutil.rmtree(self._out_dir, ignore_errors=True)
        return

    def ClampAttack(self, min_val=-0.01, max_val=0.01, only_targets=True):

        if only_targets:
            for i, (name, param) in enumerate(self._model.named_parameters()):
                if name in self._watermark_host_layers:
                    weights = torch.clamp(param, min_val, max_val)
                    param.data = weights
        else:
            for i, (name, param) in enumerate(self._model.named_parameters()):
                if 'weight' in name and 'conv' in name:
                    weights = torch.clamp(param, min_val, max_val)
                    param.data = weights

        return

    def ZeroAttack(self, min_val=-0.01, max_val=0.01, only_targets=True):

        if only_targets:
            for i, (name, param) in enumerate(self._model.named_parameters()):
                if name in self._watermark_host_layers:
                    weights = param
                    zero_tensor = torch.zeros(weights.size()).to(self._device)
                    with torch.no_grad():
                        weights = torch.where(weights > min_val, weights, zero_tensor)
                        weights = torch.where(weights < max_val, weights, zero_tensor)
                    param.data = weights
        else:
            for i, (name, param) in enumerate(self._model.named_parameters()):
                if 'convbn_2.conv.weight' in name:
                    weights = param
                    zero_tensor = torch.zeros(weights.size()).to(self._device)
                    with torch.no_grad():
                        weights = torch.where(weights > min_val, weights, zero_tensor)
                        weights = torch.where(weights < max_val, weights, zero_tensor)
                    param.data = weights

        return

    def Pruning(self, sparsity=0.1):

        parameters_to_prune = []

        for module_name, module in self._model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                parameters_to_prune.append((module, "weight"))

        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=sparsity)

        for module_name, module in self._model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.remove(module, name='weight')

        return

    def Quantization(self, q_bits=32, only_targets=True):

        def integer_quantization(W, n_bits):
            amin, amax = W.min(), torch.abs(W).max()
            delta = torch.abs(amax - amin) / 2 ** n_bits
            w_q = torch.floor(W / delta) * delta
            return w_q

        if only_targets:
            for i, (name, param) in enumerate(self._model.named_parameters()):
                if name in self._watermark_host_layers:
                    weights = param
                    with torch.no_grad():
                        weights = integer_quantization(weights, q_bits)
                    param.data = weights
        else:
            for i, (name, param) in enumerate(self._model.named_parameters()):
                if 'convbn_2.conv.weight' in name:
                    weights = param
                    zero_tensor = torch.zeros(weights.size()).to(self._device)
                    with torch.no_grad():
                        weights = integer_quantization(weights, q_bits)
                    param.data = weights

        return
