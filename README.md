![Image](Resources/vippdiism.png)

#  Chameleon Watermark: A Robust DNN Watermarking Algorithm with Invisible Watermark
A new white-box, multi-bit watermarking algorithm with strong robustness properties, including retraining for transfer learning. 
Robustness is achieved thanks to a new information coding strategy according to which the watermark message is spread across a 
number of fixed weights, whose position depends on a secret key. The weights hosting the watermark are set prior to training, 
and are left unchanged throughout the entire training procedure. The distribution of the weights carrying out the message is 
theoretically optimised to make sure that the watermarked weights are indistinguishable from the other weights, while at 
the same time keeping their amplitude as large as possible to improve robustness against retraining.

This is the implementation of the paper:
~~~
@misc{https://doi.org/10.48550/arxiv.2208.10973,
  doi = {10.48550/ARXIV.2208.10973},
  url = {https://arxiv.org/abs/2208.10973},
  author = {Tondi, Benedetta and Costanzo, Andrea and Barni, Mauro},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Cryptography and Security (cs.CR), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Robust DNN Watermarking via Fixed Embedding Weights with Optimized Distribution},
  publisher = {arXiv},
  year = {2022},
}
~~~
Download PDF from [ArXiv](https://arxiv.org/abs/2208.10973).

## Installation

Use the provided *environment.yml* to build the conda environment, then activate it:
~~~
conda env create -f environment.yml
conda activate DnnWatermarking
~~~

## Datasets

### CIFAR-10, CIFAR-100 and GTSRB

The code uses the datasets that are shipped with the torchvision package. The datasets are automatically downloaded only once (the first time that the code runs) and stored in the *data* folder.

### LSUN Horses and Cats, GAN vs real faces

For these datasets, refer to the README for the TensorFlow version of the code.
.
~~~
usage: main.py [-h] [--network {resnet18,densenet121,densenet169,xception}] [--dataset {cifar10,cifar100,gtsrb}] [--task {plan,train,embed,finetune}] [--bits BITS] [--spread SPREAD] [--c C] [--layers LAYERS] [--shares SHARES]
               [--epochs EPOCHS] [--batch-size BATCH_SIZE] --lr LR [--pretrained PRETRAINED] [--gpu GPU]
main.py: error: the following arguments are required: --lr
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
automatically loads the correct file. In this repository, we already provide the configuration files for the setups of the paper.


## Embedding the watermark 


### Planning

The code allows to plan the embedding by choosing the target layer(s) and verifying the watermark occupancy of the selected layer(s). To choose the host layer(s) run the following command:

~~~
python3 chameleondnn.py --network resnet18 --dataset cifar10 --task plan --bits 256 --spread 50 --c 1
~~~

This will print a detailed list of all the network layers, together with the shape and the total count of the weights. Here, you can pick up the host layer(s) by name from the L_Name column.
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

note that layers must be provided as a comma-separated string with their exact names. This will produce the following output, with the % occupancy in the last column W_share:

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
- PNG plots of the distribution of watermarked/non-watermarked weights for each host layer

## Finetuning / Transfer learning

To finetune a pretrained model, use the following command:

~~~
python3 chameleondnn.py --network resnet18 --dataset cifar10 --task finetune --bits 256 --spread 50 --c 1 -epochs 10 --lr 0.001 -layers "layer3.1.convbnrelu_1.conv.weight,layer4.0.convbnrelu_1.conv.weight" --pretrained models/cifar10_resnet18_B256_S50_C1.0/1/best_model.pt
~~~

This command, for example, will finetune the model trained on CIFAR-10 for 10 extra epochs on 70% of the CIFAR-10 dataset.

To perform transfer learning to a new dataset from a pretrained model, use the following command:

~~~
python3 chameleondnn.py --network resnet18 --dataset gtsrb --task finetune --bits 256 --spread 50 --c 1 -epochs 20 --lr 0.01 -layers "layer3.1.convbnrelu_1.conv.weight,layer4.0.convbnrelu_1.conv.weight" --pretrained models/cifar10_resnet18_B256_S50_C1.0/1/best_model.pt
~~~

This command, for example, will train the input pretrained model on the GTSRB dataset for 20 epochs.













