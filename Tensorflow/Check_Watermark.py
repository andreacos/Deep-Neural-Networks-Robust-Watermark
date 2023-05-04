import os
import utils
from distributions import Distribution
from CnnWatermarker import CnnWatermarker, EncodeMessage
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Search a watermark into the input Densenet model.')
    parser.add_argument('model_file',  type=str, help='The initial model file')
    args = parser.parse_args()
    model_to_test = args.model_file

    not_wm_model = 'models/checkpoints/Densenet-GTSRB-No-Watermark-ep-20/ckpt.epoch19-loss0.16.h5'

    exp_id = os.path.dirname(model_to_test).split('/')[-1]

    epochs = 20
    classes = 43
    disable_watermark = True
    run_eagerly = False
    input_size = (32, 32, 3)

    settings = model_to_test.split("/")[2].split('-')[3:]
    if settings[0] != "B":
        settings = model_to_test.split("/")[2].split('-')[5:]

    print(f"Settings: {settings}")
    B = settings[1]

    watermark_length = int(settings[1]) * int(settings[5])
    message_length = int(settings[1])

    if settings[7] == '1':
        host_layers = ['conv5_block16_1_conv']  # Single layer
    if settings[7] == '2':
        host_layers = ['conv5_block16_1_conv', 'conv5_block20_1_conv']  # Two layers
    if settings[7] == '4':
        host_layers = ['conv4_block16_1_conv', 'conv4_block20_1_conv', 'conv5_block16_1_conv', 'conv5_block20_1_conv']

    laplace_sigma_mult = float(settings[3])

    wmEmbedder = CnnWatermarker(key=123456, base_model='densenet')
    wmEmbedder.CreateNetwork(input_shape=input_size, weights="imagenet", classes=classes, host_layers=host_layers)
    watermark, message_bits = EncodeMessage(key=123456, w_length=watermark_length, msg_length=message_length)
    wmEmbedder.CreateWatermark(watermark=watermark,
                               watermark_length=watermark_length,
                               strength_distribution=Distribution(name='laplace', unwat_model=not_wm_model,
                                                                  sigma_multiplier=laplace_sigma_mult, mu=0,
                                                                  size=watermark_length, key=123456),
                               host_layers=host_layers)
    wmEmbedder.SetCustomModel(model_to_test)
    score, wrong_bits, rec_message_bits = wmEmbedder.ExtractMultibitWatermark(orig_watermark=watermark,
                                                                              message_bits=message_bits,
                                                                              host_layers=host_layers)

    # Print results to file and to terminal
    with open(os.path.join(f"{os.path.join('results', exp_id)}", "log.txt"), 'a+') as f:
        utils.print_redirect(f, f"BER = {wrong_bits / len(message_bits) * 100}%")
        utils.print_redirect(f, f"ERRORS = {wrong_bits} / {len(message_bits)}")
