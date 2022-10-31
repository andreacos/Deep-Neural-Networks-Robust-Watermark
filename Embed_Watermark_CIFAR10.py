import os
import utils
from distributions import Distribution
from CnnWatermarker import CnnWatermarker, EncodeMessage
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def host_layer_list(layer_setup='2'):
    h_layers=[]
    if layer_setup == '1':
        h_layers = ['conv5_block16_1_conv']  # Single layer
    if layer_setup == '2':
        h_layers = ['conv5_block16_1_conv', 'conv5_block20_1_conv']  # Two layers
    if layer_setup == '4':
        h_layers = ['conv4_block16_1_conv', 'conv4_block20_1_conv', 'conv5_block16_1_conv', 'conv5_block20_1_conv']

    return h_layers


if __name__ == "__main__":

    not_wm_model = 'models/checkpoints/Densenet-CIFAR10-No-watermark-ep-10/ckpt.epoch10-loss0.51.h5'

    # Watermark parameters
    disable_watermark = False
    layer_setup = '2'  # '1' or '2' or '4'
    spread = 6
    payload = 256
    C = 1

    # Network parameters
    dataset = 'cifar10'
    epochs = 10
    classes = 10
    input_size = (32, 32, 3)

    host_layers = host_layer_list(layer_setup)
    watermark_length = payload * spread

    exp_id = f"Densenet-CIFAR10-Watermarked-B-{payload}-C-{int(C)}" \
             f"-S-{int(watermark_length/payload)}-L-{len(host_layers)}-ep-{epochs}"

    wmEmbedder = CnnWatermarker(key=123456, base_model='densenet')

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
    wmEmbedder.CompileModel(run_eagerly=False, learning_rate=1e-4)

    wmEmbedder.TrainCifar(batch_size=32,
                          epochs=epochs,
                          model_dir=f"{os.path.join('models', 'checkpoints', exp_id)}",
                          run_eagerly=False,
                          output_dir=f"{os.path.join('results', exp_id)}")

    accuracy = wmEmbedder.TestCifar(output_dir=f"{os.path.join('results', exp_id)}")

    _, wrong_bits, _ = wmEmbedder.ExtractMultibitWatermark(orig_watermark=watermark,
                                                           message_bits=message_bits,
                                                           host_layers=host_layers)

    with open(os.path.join(f"{os.path.join('results', exp_id)}", "log.txt"), 'a+') as f:
        utils.print_redirect(f, exp_id)
        #for vi in variance_info:
        #    utils.print_redirect(f, vi)
        utils.print_redirect(f, f"ACC (Top 1) = {np.round(100*accuracy[0], 2)}%")
        utils.print_redirect(f, f"ACC (Top 3) = {np.round(100*accuracy[1], 2)}%")
        utils.print_redirect(f, f"TER (Top 1) = {np.round(100*(1-accuracy[0]), 2)}%")
        utils.print_redirect(f, f"TER (Top 3) = {np.round(100*(1-accuracy[1]), 2)}%")
        utils.print_redirect(f, f"BER = {wrong_bits / len(message_bits) * 100}%")
        utils.print_redirect(f, f"ERRORS = {wrong_bits} / {len(message_bits)}")
