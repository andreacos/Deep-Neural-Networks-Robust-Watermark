import os
import argparse
import torch
from Watermarker import Watermarker


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--network', default='resnet18', choices=['resnet18', 'densenet121', 'densenet169', 'xception'], help='Network')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'gtsrb'], help='Training dataset')
    parser.add_argument('--task', default='embed', choices=['plan', 'train', 'embed', 'finetune'], help='Task')
    parser.add_argument('--bits', type=int, default=256, help='Message length in bits')
    parser.add_argument('--spread', type=int, default=6, help='Watermark spread')
    parser.add_argument('--c', type=float, default=1.0, help='Watermark strength multiplier')
    parser.add_argument('--layers', type=str, default='',
                        help='Comma-separated list of target layers for watermark')
    parser.add_argument('--shares', type=str, default=None,
                        help='Comma-separated list of target layers shares for watermark')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--pretrained', type=str, help='Path to the pre-trained model (finetune only)')
    parser.add_argument('--gpu', default='1', help='System GPU index')
    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Flag to turn ON/OFF watermark embedding. Obviously ON only in Embed mode
    embed = True if args.task == "embed" else False

    # Check if the model is being retrained on the same dataset
    same_dataset = True if (args.pretrained is not None and args.pretrained != '') and args.dataset in args.pretrained else False

    shares = None
    if args.shares is not None:
        shares = [float(x) for x in args.shares.replace(" ", "").split(',')]
        assert len(args.layers.replace(" ", "").split(',')) == len(shares), "Host layers and shares havev different lengths"
        assert sum(shares) == 1, "Host layer weight shares do not sum to 1"

    if args.task == "finetune":
        assert args.pretrained != '', "Missing --pretrained model"

    # Initialize the Watermarker, the dataset and the model
    wmEmbedder = Watermarker(net=args.network, dataset=args.dataset, device=device, embed=embed, task=args.task)
    wmEmbedder.CreateDataset(batch_size=args.batch_size, same_dataset=same_dataset)
    wmEmbedder.CreateModel(input_shape=(32, 32, 3), from_model=args.pretrained)

    wmEmbedder.CreateWatermark(n_bits=args.bits,
                               spread=args.spread,
                               C=args.c,
                               host_layers=args.layers.replace(" ", "").split(','),
                               host_shares=shares)

    # Assign a unique output directory to the experiment and avoid to overwrite the same setups
    wmEmbedder.ExperimentID(args=args, same_dataset=same_dataset)

    # For Plan and Embed modes show a summary of the percentage of weights that are hosting the watermark
    if args.task in ["plan", "embed"]:
        wmEmbedder.Plan(show_target_only=False)

    # Finetune mode only: check BER before start
    if args.task in ['finetune']:
        _, ber = wmEmbedder.ReadWatermark(save_txt=True)
        print(f"BER before starting finetuning: {ber}")

    # For all modes except Plan: training and test the model
    if args.task != 'plan':
        wmEmbedder.Train(lr=args.lr, epochs=args.epochs, out_log='log.csv', save_best_only=True)
        _, test_acc = wmEmbedder.Validate(is_test=True, topk=5)

        print(f"TER TOP1: {1-test_acc[0]}")
        print(f"TER TOP5: {1-test_acc[1]}")

        # PLot target layers weights distributions
        wmEmbedder.PlotDistributions(save_data=False)

        # For all modes except Train: recover the watermark
        if args.task != 'train':
            _, ber = wmEmbedder.ReadWatermark(save_txt=True)
            print(f"BER: {ber}")

    print(f"Model and results saved in {wmEmbedder._out_dir}")
