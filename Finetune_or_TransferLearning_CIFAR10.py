import os
import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from utils import print_redirect
from CustomFitlessModel import CustomModel
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import argparse


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def change_model_output(net, num_classes):
    if net.output.shape[-1] != num_classes:
        net.Trainable = True
        x = net.layers[-2].output
        predictions = tf.keras.layers.Dense(num_classes, activation='softmax', name='new_pred_layer')(x)
        return CustomModel(inputs=net.input, outputs=predictions)
    else:
        return net


def TrainCifar(model, batch_size, epochs, model_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    (x_train, y_train), _ = cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    x_train = x_train.astype('float32')
    x_train /= 255

    data_aug = ImageDataGenerator(rotation_range=15,
                                  width_shift_range=5. / 32,
                                  height_shift_range=5. / 32, validation_split=0.1)

    # Create model directory
    os.makedirs(os.path.join(model_dir), exist_ok=True)

    mcp_save = ModelCheckpoint(
        filepath=os.path.join(model_dir, 'ckpt.epoch{epoch:02d}-loss{loss:.2f}.h5'),
        monitor='loss', verbose=1, save_weights_only=False, save_best_only=False)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                                  cooldown=0, patience=10, min_lr=0.5e-6)

    model.fit(data_aug.flow(x_train, y_train, batch_size=batch_size),
              epochs=epochs,
              callbacks=[mcp_save, reduce_lr],
              shuffle=True,
              verbose=1)


def TestCifar(model, output_dir, log_file='log.txt'):

    os.makedirs(output_dir, exist_ok=True)

    _, (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype('float32')

    predictions = model.predict(x_test / 255.0)
    y_pred = np.argmax(predictions, 1)
    y_true = np.argmax(to_categorical(y_test), axis=1)

    top1 = 0
    topN = 0
    for i in range(len(predictions)):
        i_pred = predictions[i]
        top_values = (-i_pred).argsort()[:3]
        if top_values[0] == y_true[i]:
            top1 += 1.0
        if y_true[i] in top_values:
            topN += 1.0

    top1 = top1 / len(predictions)
    topN = topN / len(predictions)

    # Print accuracy to log file and screen
    with open(os.path.join(output_dir, log_file), 'a+') as f:
        print_redirect(f, f'\n TRANSFER LEARNING TO CIFAR 10')
        print_redirect(f, f'Accuracy (top1): {np.round(100 * top1, 2)}%')
        print_redirect(f, f'Accuracy (top3): {np.round(100 * topN, 2)}%')
        print_redirect(f, f'TER (top1): {np.round(100 * (1 - top1), 2)}%')
        print_redirect(f, f'TER (top3): {np.round(100 * (1 - topN), 2)}%')

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Finetune a densenet model on CIFAR10.')
    parser.add_argument('model_file',  type=str, help='The initial model file')
    args = parser.parse_args()
    model_file = args.model_file

    task = "GTSRB" if "GTSRB" in model_file else "CIFAR10"

    exp_id = f"Densenet-CIFAR10-from-watermarked-{task}-{'-'.join(model_file.split('/')[2].split('-')[3:])}"

    n_classes = 10
    lr = 0.0001
    epochs = 10

    model = load_model(model_file, custom_objects={"CustomModel": CustomModel})
    model = change_model_output(model, n_classes)

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=lr, decay=lr / epochs),
                  metrics=['accuracy'],
                  run_eagerly=False)

    TrainCifar(model=model,
               batch_size=64,
               epochs=epochs,
               model_dir=f"{os.path.join('models', 'checkpoints', exp_id)}",
               output_dir=f"{os.path.join('results', exp_id)}")

    TestCifar(model=model,
              output_dir=f"{os.path.join('results', exp_id)}")
