![Image](Resources/vippdiism.png)

# Robust DNN Watermarking via Fixed Embedding Weights with Optimized Distribution
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

### LSUN Horses and Cats

This dataset is a subset of the [LSUN](https://www.yf.io/p/lsun) dataset, from which we chose two of the several available categories. 
We downloaded the dataset from the official repository with the [download utility](s instructed by the authors) provided by the authors. 
Recently, LSUN has been added to Tensorflow (read more [here](https://www.tensorflow.org/datasets/catalog/lsun)), which 
should make this dataset easier to download and use, however we did not use it in our experiments.

The code expects the following structure:
~~~
dataset_dir
|____ Cat
    |____ Train
    |____ Test
|____ Horse
    |____ Train
    |____ Test
~~~

Edit the ```class0_dir``` and ```class0_dir``` variables in the training scripts to point to the dataset directory.

### GAN vs real faces

We built the GAN FACES dataset as follows:
- 30.000 real faces from the [CelebAHQ](https://github.com/tkarras/progressive_growing_of_gans) dataset and 70.000 real 
faces rom the [FlickrFHQ](https://github.com/NVlabs/ffhq-dataset) dataset 
- 100.000 Gan faces created with the official release of [StyleGan2](https://github.com/NVlabs/stylegan2). 
More specifically, we used the official Docker image with model *stylegan2-ffhq-config-f.pkl* 
 (trained with FFHQ) with increasing truncation_psi parameter (10 values, 10000 images for each value). 
Use the script in create_stylegan2_images.py to reproduce these images.
- We split the two classes: 90% training and 10% test.

To train the networks for this task, we used image size 299x299x3. You can either resize all the above images 
(their default size is 1024x1024x3) or let the code resize while training/testing (slower).

~~~
dataset_dir
|____ Train
    |____ GAN
    |____ Pristine
|____ Horse
    |____ GAN
    |____ Pristine
~~~

Edit the ```class0_dir``` and ```class0_dir``` variables in the training scripts to point to the dataset directory.

## Training baseline models

Baseline, not-watermarked models for a given task are required by the watermarking algorithm to set the standard deviation 
of the Laplacian watermark strength (that is equal to the standard deviation of the target layer weights). Baseline models 
can be trained by using the same scripts used to train watermarked models, by setting the following parameter:

~~~
disable_watermark = True
~~~

This will make the code skip the watermarking while training. 

~~~
python XNET_Embed_Watermark_GANFACES.py
~~~

Once the baseline models are trained, set the ```not_wm_model = '''``` variable in each of the three scripts so that it points to
the right model path and revert to ```disable_watermark = False```.

Alternatively, [download the baseline models here](https://drive.google.com/file/d/1Ph1HQ06MGpTzO3nlamBNq2zcdyoryQ2t/view?usp=sharing) 
(450MB) and unzip them into the ```models/checkpoints``` directory.

# GANFACES

## Training an XCeption model with watermark

To train an XCeptionNet model with embedded watermark, run the script:
~~~
python XNET_Embed_Watermark_GANFACES.py 
~~~

To configure the watermark parameters, modify the following values in the script:
~~~
disable_watermark = False
layer_setup = '1'
spread = 6
payload = 256
C = 1
~~~

Set ```disable_watermark = True``` only if you want to train a (baseline) model **without watermark**.

Models for each training epoch will be saved in the following directory:

```models/checkpoints/XCeption-GANFACES-Watermarked-B-{payload}-C-{C}-S-{spread}-L-{layers}-ep-{epochs}```

Logs and results will be printed to screen and saved into a text file in the following directory:

```results/XCeption-GANFACES-Watermarked-B-{payload}-C-{C}-S-{spread}-L-{layers}-ep-{epochs}```


## Finetuning of a watermarked XCeption GANFACES model on GANFACES

To finetune a previously trained and watermarked model on GTSRB, run the following command, whose input argument is the to-be-finetuned path to the model, e.g.:

```
python XNET_Finetune_or_TransferLearning_GANFACES.py models/checkpoints/XCeption-GANFACES-Watermarked-B-256-C-1-S-6-L-1-ep-10/ckpt.epoch10-loss0.30.h5
```

Models for each training epoch will be saved in the following directory:

```models/checkpoints/XCeption-GANFACES-from-watermarked-GANFACES-B-{payload}-C-{C}-S-{spread}-L-{layers}-ep-{epochs}```

Logs and results will be printed to screen and saved into a text file in the following directory:

```results/XCeption-GANFACES-from-watermarked-GANFACES-B-{payload}-C-{C}-S-{spread}-L-{layers}-ep-{epochs}```

Note that the code will derive the watermarking setup from the directory where the model has been created (e.g.
payload=256, C-1, spread=6, layers=1). Do not move models or rename their directory at the end of the training 
(or edit the scripts to point to the correct watermarking settings).

Verify the BER of the finetuned model with the following command, whose input argument is the path to the finetuned model, e.g.:

```
python XNET_Check_Watermark.py  models/checkpoints/XCeption-GANFACES-from-watermarked-GANFACES-B-256-C-1-S-6-L-1-ep-10/ckpt.epoch10-loss0.27.h5
```

## Transfer learning from a watermarked XCeption GANFACES model to HORSECAT

To train a XCeption model on HORSECAT by starting from the weights of a previously trained and watermarked model on GANFACES, 
run the following command, whose input argument is the path to the model, e.g.:

```
python XNET_Finetune_or_TransferLearning_HORSECAT.py models/checkpoints/XCeption-GANFACES-Watermarked-B-256-C-1-S-6-L-1-ep-10/ckpt.epoch10-loss0.30.h5
```

The model will be trained on the HORSECAT dataset.  Models for each training epoch will be saved in the following directory:

```models/checkpoints/XCeption-HORSECAT-from-watermarked-GANFACES-B-{payload}-C-{C}-S-{spread}-L-{layers}-ep-{epochs}```

Logs and results will be printed to screen and saved into a text file in the following directory:

```results/XCeption-HORSECAT-from-watermarked-GANFACES-B-{payload}-C-{C}-S-{spread}-L-{layers}-ep-{epochs}```

Note that the code will derive the watermarking setup from the directory where the model has been created (e.g.
payload=256, C-1, spread=6, layers=1). Do not move models or rename their directory at the end of the training 
(or edit the scripts to point to the correct watermarking settings).

Verify the BER of the new model with the following command, whose input argument is the path to the model file, e.g.:

```
python XNET_Check_Watermark.py models/checkpoints/XCeption-HORSECAT-from-watermarked-GANFACES-B-256-C-1-S-6-L-1-ep-10/ckpt.epoch10-loss0.27.h5
```

