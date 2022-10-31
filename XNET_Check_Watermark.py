import os
import utils
from distributions import Distribution
from XNetCnnWatermarker import CnnWatermarker, EncodeMessage
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Search a watermark into the input XceptionNet model.')
    parser.add_argument('model_file',  type=str, help='The initial model file')
    args = parser.parse_args()
    model_to_test = args.model_file

    exp_id = os.path.dirname(model_to_test).split('/')[-1]

    not_wm_model = 'models/checkpoints/XCeption-GANNOGAN-No-Watermark-ep-10/ckpt.epoch09-loss0.01.h5'
    classes = 2
    disable_watermark = True
    run_eagerly = False
    input_size = (229, 229, 3)

    settings = model_to_test.split("/")[-2].split('-')[5:]
    print(f"Settings: {settings}")
    B = settings[1]

    watermark_length = int(settings[1]) * int(settings[5])
    message_length = int(settings[1])

    if settings[7] == '1':
        host_layers = ['block14_sepconv2']  # Single layer
    if settings[7] == '2':
        host_layers = ['block14_sepconv1', 'block14_sepconv2']  # Two layers
    if settings[7] == '4':
        host_layers = ['block13_sepconv1', 'block13_sepconv2', 'block14_sepconv1', 'block14_sepconv2']

    laplace_sigma_mult = float(settings[3])

    wmEmbedder = CnnWatermarker(key=123456, base_model='exception')
    wmEmbedder.CreateNetwork(input_shape=input_size, weights="imagenet", classes=classes, host_layers=host_layers)
    watermark, message_bits = EncodeMessage(key=123456, w_length=watermark_length, msg_length=message_length)
    wmEmbedder.CreateWatermark(watermark=watermark,
                               watermark_length=watermark_length,
                               strength_distribution=Distribution(name='laplace', unwat_model=not_wm_model,
                                                                  sigma_multiplier=laplace_sigma_mult, mu=0,
                                                                  size=watermark_length, key=123456),
                               host_layers=host_layers)
    wmEmbedder.SetCustomModel(model_to_test)
    _, wrong_bits, _ = wmEmbedder.ExtractMultibitWatermark(orig_watermark=watermark,
                                                           message_bits=message_bits,
                                                           host_layers=host_layers)

    # Print results to file and to terminal
    with open(os.path.join(f"{os.path.join('results', exp_id)}", "log.txt"), 'a+') as f:
        utils.print_redirect(f, f"BER = {wrong_bits / len(message_bits) * 100}%")
        utils.print_redirect(f, f"ERRORS = {wrong_bits} / {len(message_bits)}")
