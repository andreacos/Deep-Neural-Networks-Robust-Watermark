import os
import cv2
import numpy as np
from glob import glob
from CustomFitlessModel import CustomModel
import tensorflow as tf
import utils
from tensorflow.keras.models import load_model
from tqdm import tqdm
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from CustomGenerators import create_batch_generators, create_training_validation_sets, augment_on_test
import argparse


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def TrainTwoClass(model, batch_size, epochs, model_dir, class0_dir, class1_dir, labels, augmentation):
    # Create training and validation sets and labels
    x_train, x_validation, y_train, y_validation = create_training_validation_sets(class0_dir, class1_dir, labels)

    # Initialize batch generators
    training_generator, validation_generator = create_batch_generators(x_train=x_train, x_validation=x_validation,
                                                                       y_train=y_train, y_validation=y_validation,
                                                                       augmentation=augmentation)

    os.makedirs(os.path.join(model_dir), exist_ok=True)

    checkpoint_saver = ModelCheckpoint(filepath=os.path.join(model_dir, 'ckpt.epoch{epoch:02d}-loss{val_loss:.2f}.h5'),
                                       monitor='val_loss', verbose=1, save_weights_only=False, save_best_only=False)

    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=2, min_lr=0.5e-6)
    callbacks = [lr_reducer, checkpoint_saver]

    model.fit(training_generator,
              steps_per_epoch=int(len(x_train) // batch_size),
              validation_data=validation_generator,
              validation_steps=int(len(x_validation) // batch_size),
              callbacks=callbacks,
              epochs=epochs)


def TestTwoClass(model, output_dir, class0_dir, class1_dir, labels, augmentation, log_file='log.txt'):

    os.makedirs(output_dir, exist_ok=True)

    input_shape = model.input_shape[1:]

    # Image and labels
    class0_images = glob(os.path.join(class0_dir, '*.*'))
    class1_images = glob(os.path.join(class1_dir, '*.*'))

    class0_labels = [labels[0] for x in range(len(class0_images))]
    class1_labels = [labels[1] for x in range(len(class1_images))]

    test_images = class0_images + class1_images
    test_labels = class0_labels + class1_labels

    print(
        f"Found {len(test_images)} images to test ({len(class0_images)} for class 0 and {len(class1_images)} for class 1)")

    predictions = []
    pred_labels = []

    # Test
    for im_file in tqdm(test_images):

        img = cv2.imread(im_file)
        if img.shape != input_shape:
            img = cv2.resize(img, dsize=(input_shape[0], input_shape[1]))

        if augmentation:
            img = augment_on_test(img)

        # Classify
        score = model.predict(np.expand_dims(img / 255., 0))
        predictions.append(score)
        pred_labels.append(np.argmax(score, 1))

    pred_labels = np.array(pred_labels).flatten()
    test_labels = np.array(test_labels).flatten()
    lmin = min(len(pred_labels), len(test_labels))
    pred_labels = pred_labels[:lmin]
    test_labels = test_labels[:lmin]

    accuracy = np.sum(pred_labels == test_labels) / len(pred_labels)

    # Print results
    with open(os.path.join(output_dir, log_file), 'a+') as f:
        utils.print_redirect(f, f'Accuracy: {np.round(100 * accuracy, 2)}%')
        utils.print_redirect(f, f'TER: {np.round(100 * (1 - accuracy), 2)}%')

    return accuracy, (test_images, test_labels, pred_labels, predictions)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Finetune an Exception model on HORSECAT.')
    parser.add_argument('model_file',  type=str, help='The initial model file')
    args = parser.parse_args()
    model_file = args.model_file

    task = "GANFACES" if "GANFACES" in model_file else "HORSECAT"

    exp_id = f"XCeption-HORSECAT-from-watermarked-{task}-{'-'.join(model_file.split('/')[2].split('-')[3:])}"

    n_classes = 2
    input_shape = (299, 299, 3)
    lr = 1e-2
    epochs = 10

    model = load_model(model_file, custom_objects={"CustomModel": CustomModel})

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=lr, decay=lr / epochs),
                  metrics=['accuracy'], run_eagerly=False)

    TrainTwoClass(model=model,
                  batch_size=32,
                  epochs=epochs,
                  model_dir=f"{os.path.join('models', 'checkpoints', exp_id)}",
                  class0_dir='datasets/LSUN/horse/Train',
                  class1_dir='datsets/LSUN/cat/Train',
                  labels=[0, 1],
                  augmentation=False)

    TestTwoClass(model=model,
                 class0_dir='datasets/LSUN/horse/Test',
                 class1_dir='datsets/LSUN/cat/Test',
                 labels=[0, 1],
                 augmentation=False,
                 output_dir=f"{os.path.join('results', exp_id)}")
