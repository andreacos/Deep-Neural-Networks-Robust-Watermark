import os
import argparse
import numpy as np
import _pickle as pickle
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from collections import defaultdict
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import gtsrb_subset, print_redirect
from CustomFitlessModel import CustomModel


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def change_model_output(net, num_classes):
    if net.output.shape[-1] != num_classes:
        net.Trainable = True
        x = net.layers[-2].output
        predictions = tf.keras.layers.Dense(num_classes, activation='softmax', name='new_pred_layer')(x)
        return CustomModel(inputs=net.input, outputs=predictions)
    else:
        return net


def TrainGTSRB(model, batch_size, epochs, model_dir, gtsrdb_dir, output_dir, subset=None, cutoff=0.7):

    os.makedirs(output_dir, exist_ok=True)

    (trainX, trainY) = pickle.load(open(os.path.join(gtsrdb_dir, 'Train.p'), 'rb'))
    (testX, testY) = pickle.load(open(os.path.join(gtsrdb_dir, 'Test.p'), 'rb'))

    trainX = trainX.astype("float32") / 255.0
    testX = testX.astype("float32") / 255.0

    num_labels = len(np.unique(trainY))
    if subset is not None:
        trainX, trainY, testX, testY = gtsrb_subset(trainX, trainY, testX, testY, cutoff)

    trainY = to_categorical(trainY, num_labels)
    testY = to_categorical(testY, num_labels)

    class_totals = trainY.sum(axis=0)
    class_weight = class_totals.max() / class_totals
    keys = np.arange(num_labels)

    class_weights = defaultdict(list)
    for k, v in zip(keys, class_weight):
        class_weights[k].append(v)

    data_aug = ImageDataGenerator(rotation_range=10, zoom_range=0.15, width_shift_range=0.1, height_shift_range=0.1,
                                  shear_range=0.15, horizontal_flip=False, vertical_flip=False)

    # Create model directory
    os.makedirs(os.path.join(model_dir), exist_ok=True)

    mcp_save = ModelCheckpoint(
        filepath=os.path.join(model_dir, 'ckpt.epoch{epoch:02d}-loss{val_loss:.2f}.h5'),
        monitor='val_loss', verbose=1, save_weights_only=False, save_best_only=False)

    model.fit(data_aug.flow(trainX, trainY, batch_size=batch_size),
              epochs=epochs,
              validation_data=(testX, testY),
              class_weight=class_weights,
              callbacks=[mcp_save],
              verbose=1)


def TestGTSRB(model, gtsrdb_dir, output_dir, log_file='log.txt'):

    os.makedirs(output_dir, exist_ok=True)

    (testX, testY) = pickle.load(open(os.path.join(gtsrdb_dir, 'Test.p'), 'rb'))

    # Prepare for test
    testX = testX.astype("float32") / 255.0
    num_labels = len(np.unique(testY))
    testY = to_categorical(testY, num_labels)

    # Predict
    predictions = model.predict(testX)
    y_true = np.argmax(testY, axis=1)
    y_pred = np.argmax(predictions, axis=1)

    # Print accuracy to log file and screen
    with open(os.path.join(output_dir, log_file), 'a+') as f:
        print_redirect(f, f'\n FINETUNING ON GTSRB')
        print_redirect(f, f' Test Accuracy: {np.round(100 * np.sum(y_pred == y_true) / len(y_true), 2)}%')
        print_redirect(f, f' TER: {np.round(100 * (1 - np.sum(y_pred == y_true) / len(y_true)), 2)}%')
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Finetune a densenet model on GTSRB.')
    parser.add_argument('model_file',  type=str, help='The initial model file')
    args = parser.parse_args()
    model_file = args.model_file

    exp_id = f"Densenet-GTSRB-from-watermarked-GTSRB-{'-'.join(model_file.split('/')[2].split('-')[3:])}"

    n_classes = 43
    lr = 0.001
    epochs = 10

    model = load_model(model_file, custom_objects={"CustomModel": CustomModel})
    model = change_model_output(model, n_classes)

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=lr, decay=lr / epochs),
                  metrics=['accuracy'], run_eagerly=False)

    TrainGTSRB(model=model,
               subset=None,
               gtsrdb_dir='datasets/GTSRB',
               batch_size=64,
               epochs=epochs,
               model_dir=f"{os.path.join('models', 'checkpoints', exp_id)}",
               output_dir=f"{os.path.join('results', exp_id)}")

    TestGTSRB(model=model,
              gtsrdb_dir='datasets/GTSRB',
              output_dir=f"{os.path.join('results', exp_id)}")
