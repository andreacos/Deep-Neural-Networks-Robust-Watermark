![Image](Resources/vippdiism.png)

#  Robust DNN Watermarking via Fixed Embedding Weights with Optimized Distribution

![Image](Resources/chameleon.png)

A new white-box, multi-bit watermarking algorithm with strong robustness properties, including retraining for transfer learning. 
Robustness is achieved thanks to a new information coding strategy according to which the watermark message is spread across a 
number of fixed weights, whose position depends on a secret key. The weights hosting the watermark are set prior to training, 
and are left unchanged throughout the entire training procedure. The distribution of the weights carrying out the message is 
theoretically optimised to make sure that the watermarked weights are indistinguishable from the other weights, while at 
the same time keeping their amplitude as large as possible to improve robustness against retraining.

This is the implementation of the paper:
~~~
@misc{tondi2022robust,
      title={Robust DNN Watermarking via Fixed Embedding Weights with Optimized Distribution}, 
      author={Benedetta Tondi and Andrea Costanzo and Mauro Barni},
      year={2022},
      eprint={2208.10973},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
~~~
Download PDF from [ArXiv](https://arxiv.org/abs/2208.10973).


## Additional material

In the tables below, we report all the information on the host layers ```k```, the total number of weights in the layer (```Nk```), and the variance of the distribution of the non-watermarked weights for each layer (```σk2```), 
that for the sake of brevity were not reported in the paper. The "x" in the columns *Layers = N* indicates that the corresponding layer is used in that setup. Note that *Layer depth* refers to the index of the host layer according to PyTorch.

ResNet18:

| Layer                      | Layer depth | Shape            | Weights   | Variance | Standard dev. | Layers = 2 | Layers = 4 | Layers = 8 |
| -------------------------- | ----------- | ---------------- | --------- | -------- | ------------- | ---------- | ---------- | ---------- |
| layer3.0.convbnrelu_1.conv | 30/60       | [256, 128, 3, 3] | 294.912   | 0,00087  | 0,02944       |            |            | x          |
| layer3.0.convbn_2.conv     | 33/60       | [256, 256, 3, 3] | 589.824   | 0,00087  | 0,02943       |            | x          | x          |
| layer3.1.convbnrelu_1.conv | 39/60       | [256, 256, 3, 3] | 589.824   | 0,00087  | 0,02942       |            |            | x          |
| layer3.1.convbn_2.conv     | 42/60       | [256, 256, 3, 3] | 589.824   | 0,00087  | 0,0295        | x          | x          | x          |
| layer4.0.convbnrelu_1.conv | 45/60       | [512, 256, 3, 3] | 1.179.648 | 0,00043  | 0,02085       |            |            | x          |
| layer4.0.convbn_2.conv     | 48/60       | [512, 512, 3, 3] | 2.359.296 | 0,00043  | 0,02083       | x          | x          | x          |
| layer4.1.convbnrelu_1.conv | 54/60       | [512, 512, 3, 3] | 2.359.296 | 0,00043  | 0,02085       |            |            | x          |
| layer4.1.convbn_2.conv     | 57/60       | [512, 512, 3, 3] | 2.359.296 | 0,00043  | 0,02081       |            | x          | x          |

DenseNet169:

| Layer           | Layer depth | Shape             | Weights | Variance | Standard dev. | Layers = 2 | Layers = 4 | Layers = 6 | Layers = 12 |
| --------------- | ----------- | ----------------- | ------- | -------- | ------------- | ---------- | ---------- | ---------- | ----------- |
| dense4.20.conv1 | 432/504      | [128, 1280, 1, 1] | 163.840 | 0,00026  | 0,01612       |            |            |            | x           |
| dense4.21.conv1 | 438/504     | [128, 1312, 1, 1] | 167.936 | 0,00025  | 0,01591       |            |            |            | x           |
| dense4.22.conv1 | 444/504     | [128, 1344, 1, 1] | 172.032 | 0,00025  | 0,01575       |            |            |            | x           |
| dense4.23.conv1 | 450/504     | [128, 1376, 1, 1] | 176.128 | 0,00024  | 0,01559       |            |            |            | x           |
| dense4.24.conv1 | 456/504     | [128, 1408, 1, 1] | 180.224 | 0,00024  | 0,01541       |            |            |            | x           |
| dense4.25.conv1 | 462/504     | [128, 1440, 1, 1] | 184.320 | 0,00023  | 0,01521       |            |            |            | x           |
| dense4.26.conv1 | 468/504     | [128, 1472, 1, 1] | 188.416 | 0,00023  | 0,01505       |            | x          | x          | x           |
| dense4.27.conv1 | 474/504     | [128, 1504, 1, 1] | 192.512 | 0,00022  | 0,01489       |            | x          | x          | x           |
| dense4.28.conv1 | 480/504     | [128, 1536, 1, 1] | 196.608 | 0,00022  | 0,01474       | x          | x          | x          | x           |
| dense4.29.conv1 | 486/504     | [128, 1568, 1, 1] | 200.704 | 0,00021  | 0,01457       | x          | x          | x          | x           |
| dense4.30.conv1 | 492/504     | [128, 1600, 1, 1] | 204.800 | 0,00021  | 0,01441       |            |            | x          | x           |
| dense4.31.conv1 | 498/504     | [128, 1632, 1, 1] | 208.896 | 0,0002   | 0,01427       |            |            | x          | x           |


## Installation

Use the provided *environment.yml* to build the conda environment, then activate it:
~~~
conda env create -f environment.yml
conda activate cuml-torch
~~~

## Datasets

### CIFAR-10, CIFAR-100 and GTSRB

The code uses the datasets that are shipped with the torchvision package. The datasets are automatically downloaded only once (the first time) and stored in the *data* folder.

### LSUN Horses and Cats, GAN vs real faces

For these datasets, refer to the [README of TensorFlow version](https://github.com/andreacos/Deep-Neural-Networks-Robust-Watermark/blob/main/README_xception.md).

## Command line arguments

The code accepts the following input arguments:
~~~
usage: chameleondnn.py [-h] [--network {resnet18,densenet121,densenet169,xception}] [--dataset {cifar10,cifar100,gtsrb}] [--task {plan,train,embed,finetune}] [--bits BITS] [--spread SPREAD] [--c C] [--layers LAYERS] [--shares SHARES]
               [--epochs EPOCHS] [--batch-size BATCH_SIZE] --lr LR [--pretrained PRETRAINED] [--gpu GPU]
~~~

## Training baseline models

Baseline, non-watermarked models for a given task are required by the watermarking algorithm to set the standard deviation of the Laplacian watermark strength (that is equal to the standard deviation of the target layer weights). 
Baseline models can be trained with the following command:

~~~
python3 chameleondnn.py --network resnet18 --dataset cifar10 --task train --epochs 200 --lr 0.01
~~~

where:
-  ```network``` can be one value in ```(resnet18,  densenet169,  xception,  densenet121)```
-  ```dataset``` can be one value in ```(cifar10,  cifar100,  gtsrb)```

At the end of the training, a json file for the selected network and dataset is created in the *config* folder (e.g., *config/config-resnet18-cifar10.json*). The values in the file are used by the embedding procedure, which
automatically loads the correct file. In this repository, we already provide the configuration files for the setups discussed in the paper.


## Embedding the watermark 


### Planning

The code allows to plan the embedding phase by choosing the target layer(s) and verifying the watermark occupancy in the selected layer(s). To choose the host layer(s) run the following command:

~~~
python3 chameleondnn.py --network resnet18 --dataset cifar10 --task plan --bits 256 --spread 50 --c 1
~~~

This will print a detailed list of all the network layers, together with the shape and the total count of their weights. Here, you can pick up the host layer(s) by name from the *L_Name* column.
For example: 

~~~
L_index  L_Name                             L_shape                         L_count    L_variance    L_std_dev  W_spread    W_mgs    W_wat    W_host    W_share
---------  ---------------------------------  ----------------------------  ---------  ------------  -----------  ----------  -------  -------  --------  ---------
...
       39  layer3.1.convbnrelu_1.conv.weight  torch.Size([256, 256, 3, 3])     589824   0.000866249    0.0294321  nd          nd       nd
       40  layer3.1.convbnrelu_1.bn.weight    torch.Size([256])                   256   0              0          nd          nd       nd
       42  layer3.1.convbn_2.conv.weight      torch.Size([256, 256, 3, 3])     589824   0.000868995    0.0294787  nd          nd       nd
       43  layer3.1.convbn_2.bn.weight        torch.Size([256])                   256   0              0          nd          nd       nd
       45  layer4.0.convbnrelu_1.conv.weight  torch.Size([512, 256, 3, 3])    1179648   0.000433913    0.0208306  nd          nd       nd
...
~~~

Then, to verify the watermark occupancy, run the following command:

~~~
python3 chameleondnn.py --network resnet18 --dataset cifar10 --task plan --bits 256 --spread 50 --c 1 --layers "layer3.1.convbnrelu_1.conv.weight,layer4.0.convbnrelu_1.conv.weight"
~~~

note that layers must be provided as a comma-separated string with their exact names. This will produce the following output, with the % occupancy in the last column *W_share*:

~~~
L_index  L_Name                             L_shape                         L_count    L_variance    L_std_dev  W_spread    W_mgs    W_wat    W_host    W_share
---------  ---------------------------------  ----------------------------  ---------  ------------  -----------  ----------  -------  -------  --------  ---------
...
       39  layer3.1.convbnrelu_1.conv.weight  torch.Size([256, 256, 3, 3])     589824   0.000868523    0.0294707          50      256    12800  *         1.09% (6400)
       45  layer4.0.convbnrelu_1.conv.weight  torch.Size([512, 256, 3, 3])    1179648   0.000434388    0.020842           50      256    12800  *         0.54% (6400)
...
~~~

If you are satisfied with the setup, you can proceed to watermark embedding.


### Embedding

To train a model with watermark embedding, use the following command:

~~~
python3 chameleondnn.py --network resnet18 --dataset cifar10 --task embed --bits 256 --spread 50 --c 1 -epochs 200 --lr 0.01 -layers "layer3.1.convbnrelu_1.conv.weight,layer4.0.convbnrelu_1.conv.weight"
~~~

The trained model, training history, metrics, input arguments summary and optional output data will be stored in the folder *models/cifar10_resnet18_B256_S50_C1.0/1*. If more than one experiment is carried out with the same setup (e.g., by changing learning rate, epochs or host layers), the code will create new folders incrementally: *models/cifar10_resnet18_B256_S50_C1.0/2*, *models/cifar10_resnet18_B256_S50_C1.0/3* and so on.
The output directory contains the following data:
- *best_model.pt*: the trained model weights (save best only)
- *args.txt*: the list of input arguments
- *log.csv*: the training history
- *plan.csv*: the output of the planning procedure with % of watermark occupancy (if planning was actually carried out)
- *results.txt*: a file containing accuracy, TER and BER.
- PNG plots for the distribution of watermarked/non-watermarked weights for each host layer

## Finetuning

To finetune a pretrained model, use the following command:

~~~
python3 chameleondnn.py --network resnet18 --dataset cifar10 --task finetune --bits 256 --spread 50 --c 1 -epochs 10 --lr 0.001 -layers "layer3.1.convbnrelu_1.conv.weight,layer4.0.convbnrelu_1.conv.weight" --pretrained models/cifar10_resnet18_B256_S50_C1.0/1/best_model.pt
~~~

This command, for example, will finetune the model trained on CIFAR-10 for 10 extra epochs on 70% of the CIFAR-10 dataset. Output data will be stored in the folder *models/gtsrb_resnet18_incremental_B256_S50_C1.0/1*.

##  Transfer learning

To perform transfer learning to a new dataset from a pretrained model, use the following command:

~~~
python3 chameleondnn.py --network resnet18 --dataset gtsrb --task finetune --bits 256 --spread 50 --c 1 -epochs 20 --lr 0.01 -layers "layer3.1.convbnrelu_1.conv.weight,layer4.0.convbnrelu_1.conv.weight" --pretrained models/cifar10_resnet18_B256_S50_C1.0/1/best_model.pt
~~~

This command, for example, will train the input pretrained model on the GTSRB dataset for 20 epochs. Output data will be stored in the folder *models/gtsrb_resnet18_finetune_B256_S50_C1.0/1*.













