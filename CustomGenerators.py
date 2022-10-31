import os
import cv2
import math
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from utils import rotate, crop_around_center, largest_rotated_rect, jpeg_compression_in_buffer
from utils import random_jpeg_compression, random_flip, random_resize, random_rotation
import random


def augment_on_test(x):
    return augment_on_test_w_params(x, width=299,
                                    jpeg_probability=50,
                                    jpeg_range=np.arange(70, 99, 1),
                                    resize_probability=80,
                                    resize_range=np.arange(0.8, 1.3, 0.1),
                                    flip_probability=20,
                                    rotation_probability=30,
                                    rotation_range=np.arange(-15, 15, 1))


def augment_on_test_w_params(x, width, jpeg_probability, jpeg_range, rotation_probability, rotation_range,
                             flip_probability, resize_probability, resize_range):
    """

    :param x:
    :param width:
    :param jpeg_probability:
    :param jpeg_range:
    :param rotation_probability:
    :param rotation_range:
    :param flip_probability:
    :param resize_probability:
    :param resize_range:
    :return:
    """

    do_jpeg = np.random.randint(0, 101)
    do_rotate = np.random.randint(0, 101)
    do_flip = np.random.randint(0, 101)
    do_resize = np.random.randint(0, 101)

    # Random JPEG compression
    if do_jpeg <= jpeg_probability:
        x = random_jpeg_compression(x, jpeg_range=jpeg_range)

    # Random rotation
    elif do_rotate <= rotation_probability:
        x = random_rotation(x, rotation_range=rotation_range, width=width)

    # Random flip
    elif do_flip <= flip_probability:
        x = random_flip(x)

    # Random resize
    if do_resize <= resize_probability:
        x = random_resize(x, resize_range=resize_range, width=width)

    return x


def create_training_validation_sets(class0_dir, class1_dir, labels, cutoff=None):

    filenames_train_0 = glob(os.path.join(class0_dir, '*.*'))
    filenames_train_0 = [x for x in filenames_train_0 if 'print' not in x and 'nvidia' not in x]

    if cutoff is not None:
        random.seed(1234)
        random.shuffle(filenames_train_0)
        filenames_train_0 = filenames_train_0[:int(len(filenames_train_0*cutoff))]

    filenames_train_1 = glob(os.path.join(class1_dir, '*.*'))
    filenames_train_1 = [x for x in filenames_train_1 if 'print' not in x and 'nvidia' not in x ]

    if cutoff is not None:
        random.seed(1234)
        random.shuffle(filenames_train_1)
        filenames_train_1 = filenames_train_1[:int(len(filenames_train_1*cutoff))]

    filenames_train = filenames_train_0 + filenames_train_1
    labels_train = [labels[0] for i in filenames_train_0] + [labels[1] for i in filenames_train_1]

    x_train, x_validation, y_train, y_validation = train_test_split(filenames_train, labels_train,
                                                                    test_size=0.025, random_state=42)
    y_train = tf.keras.utils.to_categorical(y_train, len(labels))
    y_validation = tf.keras.utils.to_categorical(y_validation, len(labels))

    return x_train, x_validation, y_train, y_validation


def create_cifar_generators(x_train, x_validation, x_test):

    train_generator = ImageDataGenerator(rescale=1./255)#, rotation_range=2, horizontal_flip=True, zoom_range=.1,
                                         #width_shift_range=0.1, height_shift_range=0.1)
    val_generator = ImageDataGenerator(rescale=1./255)#, rotation_range=2, horizontal_flip=True, zoom_range=.1,
                                       #width_shift_range=0.1, height_shift_range=0.1)
    test_generator = ImageDataGenerator(rescale=1./255)#, rotation_range=2, horizontal_flip=True, zoom_range=.1,
                                       # width_shift_range=0.1, height_shift_range=0.1)
    train_generator.fit(x_train)
    val_generator.fit(x_validation)
    test_generator.fit(x_test)

    return train_generator, val_generator, test_generator


def create_food101_generators(train_dir, test_dir, batch_size, target_size):

    train_datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.125,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.125,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=True,  # randomly flip images
                rescale=1. / 255,
                fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=target_size,
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(test_dir,
                                                      target_size=target_size,
                                                      batch_size=batch_size,
                                                      class_mode='categorical',
                                                      shuffle=False)

    return train_generator, test_generator


def create_batch_generators(x_train, x_validation, y_train, y_validation, augmentation):

    train_generator = BatchGenerator(x_train, y_train, 16,
                                     scale=1. / 255,
                                     width=299,
                                     augmentation=augmentation,
                                     jpeg_prob=50,
                                     jpeg_factors=np.arange(70, 99, 1),
                                     resize_prob=80,
                                     resize_factors=np.arange(0.8, 1.3, 0.1),
                                     flip_prob=20,
                                     rotation_prob=30,
                                     rotation_range=np.arange(-15, 15, 1))

    val_generator = BatchGenerator(x_validation, y_validation, 16,
                                   scale=1. / 255,
                                   width=299,
                                   augmentation=augmentation,
                                   jpeg_prob=50,
                                   jpeg_factors=np.arange(70, 99, 5),
                                   resize_prob=80,
                                   resize_factors=np.arange(0.8, 1.3, 0.1),
                                   flip_prob=20,
                                   rotation_prob=30,
                                   rotation_range=np.arange(-15, 15, 1))

    print(f"Augmentation is {str(augmentation).upper()}")

    return train_generator, val_generator


class BatchGenerator(tf.keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size, width, scale, augmentation, jpeg_prob, jpeg_factors,
                 resize_prob, resize_factors, flip_prob, rotation_prob, rotation_range):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.width = width
        self.scale_factor = scale
        self.enable_augmentation = augmentation
        self.jpeg_probability = jpeg_prob
        self.jpeg_range = jpeg_factors
        self.resize_probability = resize_prob
        self.resize_range = resize_factors
        self.flip_probability = flip_prob
        self.rotation_probability = rotation_prob
        self.rotation_range = rotation_range

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        arr = []
        for file_name in batch_x:

            im = cv2.imread(str(file_name))

            # Augment data if augmentation is enable
            if im.shape != (self.width, self.width):
                im = cv2.resize(im, dsize=(self.width, self.width))

            if self.enable_augmentation:
                im = self.augment(im)

            arr.append(im)

        return np.array(arr) * self.scale_factor, np.array(batch_y)

    def augment(self, x):
        """
        Random augmentations
        :param x: input image
        :return: augmented image
        """

        do_jpeg = np.random.randint(0, 101)
        do_rotate = np.random.randint(0, 101)
        do_flip = np.random.randint(0, 101)
        do_resize = np.random.randint(0, 101)

        # Random JPEG compression
        if do_jpeg <= self.jpeg_probability:
            x = self.random_jpeg_compression(x)

        # Random rotation
        elif do_rotate <= self.rotation_probability:
            x = self.random_rotation(x)

        # Random flip
        elif do_flip <= self.flip_probability:
            x = self.random_flip(x)

        # Random resize
        if do_resize <= self.resize_probability:
            x = self.random_resize(x)

        return x

    def random_jpeg_compression(self, x):
        """
        JPEG compression in file stream with random quality
        :param x: input image
        :return: JPEG compressed image
        """
        qf = np.random.choice(self.jpeg_range)
        return np.array(jpeg_compression_in_buffer(x, int(qf)))

    def random_resize(self, x):
        """
        Resize with random scale factor f. If f<1 the image is resized to the training size. If f>1 the image is cropped
        from the center to the training size
        :param x: input image
        :return: resized image
        """

        scale_f = np.random.choice(self.resize_range)

        x = cv2.resize(x, dsize=None, fx=scale_f, fy=scale_f, interpolation=cv2.INTER_CUBIC)

        if scale_f < 1:
            x = cv2.resize(x, dsize=(self.width, self.width), interpolation=cv2.INTER_CUBIC)

        elif scale_f > 1:
            center = (x.shape[0]//2, x.shape[1]//2)
            x = x[center[0]-math.ceil(self.width/2.):center[0]+math.floor(float(self.width)/2.),
                  center[1]-math.ceil(self.width/2.):center[1]+math.floor(self.width/2.)]
        return x

    def random_flip(self, x):
        """
        Randomly flip image (left-right or up-down)
        :param x: input image
        :return: flipped image
        """
        # Randomly choose to flip rows (np.fliplr) or column (np.flipud)
        return np.flip(x, np.random.randint(2))

    def random_rotation(self, x):
        """
        Image rotation with random angle. Black borders are cropped and cropped image is resize to training size
        :param x: input image
        :return: rotated image
        """
        angle = np.random.choice(self.rotation_range)
        x = rotate(x, angle)
        x = crop_around_center(x, *largest_rotated_rect(self.width, self.width, math.radians(angle)))
        x = cv2.resize(x, dsize=(self.width, self.width), interpolation=cv2.INTER_CUBIC)
        return x
