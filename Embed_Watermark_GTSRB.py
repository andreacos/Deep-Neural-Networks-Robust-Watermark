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

    not_wm_model = 'models/checkpoints/Densenet-GTSRB-No-Watermark-ep-20/ckpt.epoch19-loss0.16.h5'

    # Watermark parameters
    disable_watermark = False
    layer_setup = '2'   # '1' or '2' or '4'
    spread = 12
    payload = 256
    C = 1

    # Network parameters
    epochs = 10
    classes = 43
    input_size = (32, 32, 3)

    watermark_length = payload * spread
    host_layers = host_layer_list(layer_setup)

    exp_id = f"Densenet-GTSRB-Watermarked-B-{payload}-C-{C}" \
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
    wmEmbedder.CompileModel(run_eagerly=False, learning_rate=0.001)

    wmEmbedder.TrainGtsrdb(gtsrb_dir='datasets/GTSRB',
                           batch_size=64,
                           epochs=epochs,
                           model_dir=f"{os.path.join('models', 'checkpoints', exp_id)}",
                           output_dir=f"{os.path.join('results', exp_id)}",
                           run_eagerly=False)

    accuracy = wmEmbedder.TestGtsrdb(gtsrb_dir='datasets/GTSRB',
                                     output_dir=f"{os.path.join('results', exp_id)}")

    _, wrong_bits, _ = wmEmbedder.ExtractMultibitWatermark(orig_watermark=watermark,
                                                           message_bits=message_bits,
                                                           host_layers=host_layers)

    # Print results to file and to terminal
    with open(os.path.join(f"{os.path.join('results', exp_id)}", "log.txt"), 'a+') as f:
        utils.print_redirect(f, exp_id)
        # for vi in variance_info:
        #     utils.print_redirect(f, vi)
        utils.print_redirect(f, f"ACC = {np.round(accuracy, 2)}%")
        utils.print_redirect(f, f"TER = {np.round(100-accuracy, 2)}%")
        utils.print_redirect(f, f"BER = {wrong_bits / len(message_bits) * 100}%")
        utils.print_redirect(f, f"ERRORS = {wrong_bits} / {len(message_bits)}")
