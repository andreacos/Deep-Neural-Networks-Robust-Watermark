
import os
import utils
from distributions import Distribution
from XNetCnnWatermarker import CnnWatermarker, EncodeMessage
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def host_layer_list(layer_setup='2'):
    h_layers=[]
    if layer_setup == '1':
        h_layers = ['block14_sepconv1']  # Single layer
    if layer_setup == '2':
        h_layers = ['block14_sepconv1', 'block14_sepconv2']  # Two layers
    if layer_setup == '4':
        h_layers = ['block13_sepconv1', 'block13_sepconv2', 'block14_sepconv1', 'block14_sepconv2']

    return h_layers


if __name__ == "__main__":

    not_wm_model = 'models/checkpoints/XCeption-GANNOGAN-No-Watermark-ep-10/ckpt.epoch09-loss0.01.h5'

    # Watermark parameters
    disable_watermark = False
    layer_setup = '1'
    spread = 12
    payload = 256
    C = 1

    # Network parameters
    epochs = 10
    classes = 2
    input_size = (299, 299, 3)

    host_layers = host_layer_list(layer_setup)
    watermark_length = payload * spread

    exp_id = f"XCeption-GANFACES-Watermarked-B-{payload}-C-{C}" \
             f"-S-{int(watermark_length/payload)}-L-{len(host_layers)}-ep-{epochs}"

    wmEmbedder = CnnWatermarker(key=123456, base_model='exception')

    wmEmbedder.CreateNetwork(input_shape=input_size, weights="imagenet", classes=classes, host_layers=host_layers)
    watermark, message_bits = EncodeMessage(key=123456, w_length=watermark_length, msg_length=payload)
    variance_info = wmEmbedder.CreateWatermark(watermark=watermark,
                                               watermark_length=watermark_length,
                                               strength_distribution=Distribution(name='laplace',
                                                                                  unwat_model=not_wm_model,
                                                                                  sigma_multiplier=C,
                                                                                  mu=0,
                                                                                  size=watermark_length, key=123456),
                                               host_layers=host_layers)

    wmEmbedder.DisableEmbedding(disabled=disable_watermark)
    wmEmbedder.CompileModel(run_eagerly=False, learning_rate=1e-2)

    wmEmbedder.Train(class0_dir="datasets/GANFACES/Train/Pristine",
                     class1_dir="datasets/GANFACES/Train/GAN",
                     labels=[0, 1],
                     batch_size=32,
                     epochs=epochs,
                     augmentation=False,
                     model_dir=f"{os.path.join('models', 'checkpoints', exp_id)}",
                     run_eagerly=False)

    accuracy = wmEmbedder.Test(model_path=None,
                               class0_dir="datasets/GANFACES/Test/Pristine",
                               class1_dir="datasets/GANFACES/Test/GAN",
                               labels=[0, 1],
                               augmentation=False,
                               output_dir=f"{os.path.join('results', exp_id)}")

    _, wrong_bits, _ = wmEmbedder.ExtractMultibitWatermark(orig_watermark=watermark,
                                                           message_bits=message_bits,
                                                           host_layers=host_layers)

    # Print results to file and to terminal
    with open(os.path.join(f"{os.path.join('results', exp_id)}", "log.txt"), 'a+') as f:
        utils.print_redirect(f, exp_id)
        #for vi in variance_info:
        #    utils.print_redirect(f, vi)
        utils.print_redirect(f, "RESULTS FOR TRAINING WATERMARKED GANFACES MODEL")
        utils.print_redirect(f, f"ACC = {np.round(100*accuracy[0], 2)}%")
        utils.print_redirect(f, f"TER = {np.round(100*(1-accuracy[0]), 2)}%")
        utils.print_redirect(f, f"BER = {wrong_bits / len(message_bits) * 100}%")
        utils.print_redirect(f, f"ERRORS = {wrong_bits} / {len(message_bits)}")
