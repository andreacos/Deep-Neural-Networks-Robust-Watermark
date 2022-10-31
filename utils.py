import random
import cv2
import sys
import h5py
from tqdm import tqdm
import numpy as np
from PIL import Image
import math
import glob
import os
from io import BytesIO
import imageio
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

np.set_printoptions(precision=2)


def lorem_ipsum(length=256, bit_encode=8):

    lorem = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore " \
            "et dolore magna aliqua Ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut " \
            "aliquip ex ea commodo consequat Duis aute irure dolor in reprehenderit in voluptate velit esse " \
            "cillum dolore eu fugiat nulla pariatur Excepteur sint occaecat cupidatat non proident sunt in culpa " \
            "qui officia deserunt mollit anim id est laborum"

    return lorem[:int(length/bit_encode)]


def tobits(s):
    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result


def frombits(bits):
    chars = []
    for b in range(int(len(bits) / 8)):
        byte = bits[b*8:(b+1)*8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)


def print_redirect(file, *args):
    temp = sys.stdout # assign console output to a variable
    print(' '.join([str(arg) for arg in args]) )
    sys.stdout = file
    print(' '.join([str(arg) for arg in args]))
    sys.stdout = temp # set stdout back to console output


def contains_duplicate(input):

    # Cast each row (list) into tuple
    input = map(tuple, input)
    has_duplicates = False

    # now create dictionary and print all rows having frequency greater than 1
    freqDict = Counter(input)
    for (row, freq) in freqDict.items():
        if freq > 1:
            print(row)
            has_duplicates = True

    return has_duplicates


def euclid_dist(t1, t2):
    return np.sqrt(((t1 - t2) ** 2).sum(axis=1))


def random_antipodal_code(batch, dim, key, max_attempts=10):

    attempts = 0
    np.random.seed(key)
    x = np.random.uniform(-2, 2, (batch, int(dim)))
    while contains_duplicate(x) and attempts <= max_attempts:
        attempts += 1
        x = np.random.uniform(-2, 2, (batch, int(dim)))

    if attempts > max_attempts:
        raise AssertionError(f"Unable to generate {batch} random antipodal sequences of length {dim}.")

    return np.int8(np.sign(x))


def print_supported_layers():
    return f'{str()}, {str()}'


def generate_pairs(xvals, yvals):

    pairs = []
    for x in xvals:
        for y in yvals:
            pairs.append((x, y))
    return pairs


def jpeg_compression_in_buffer(x_in, jpg_quality):

    x = Image.fromarray(np.uint8(x_in))

    buf = BytesIO()
    x.save(buf, format='jpeg', quality=jpg_quality)

    with BytesIO(buf.getvalue()) as stream:
        x_jpg = imageio.imread(stream)

    return x_jpg


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def largest_rotated_rect(w, h, angle):

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if width > image_size[0]:
        width = image_size[0]

    if height > image_size[1]:
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def random_jpeg_compression(x, jpeg_range):
    """
    JPEG compression in file stream with random quality
    :param x: input image
    :return: JPEG compressed image
    """
    qf = np.random.choice(jpeg_range)
    return np.array(jpeg_compression_in_buffer(x, int(qf)))


def random_resize(x, width, resize_range):
    """
    Resize with random scale factor f. If f<1 the image is resized to the training size. If f>1 the image is cropped
    from the center to the training size
    :param x: input image
    :return: resized image
    """

    scale_f = np.random.choice(resize_range)

    x = cv2.resize(x, dsize=None, fx=scale_f, fy=scale_f, interpolation=cv2.INTER_CUBIC)

    if scale_f < 1:
        x = cv2.resize(x, dsize=(width, width), interpolation=cv2.INTER_CUBIC)

    elif scale_f > 1:
        center = (x.shape[0]//2, x.shape[1]//2)
        x = x[center[0]-math.ceil(width/2.):center[0]+math.floor(float(width)/2.),
              center[1]-math.ceil(width/2.):center[1]+math.floor(width/2.)]
    return x


def random_flip(x):
    """
    Randomly flip image (left-right or up-down)
    :param x: input image
    :return: flipped image
    """
    # Randomly choose to flip rows (np.fliplr) or column (np.flipud)
    return np.flip(x, np.random.randint(2))


def random_rotation(x, width, rotation_range):
    """
    Image rotation with random angle. Black borders are cropped and cropped image is resize to training size
    :param x: input image
    :return: rotated image
    """
    angle = np.random.choice(rotation_range)
    x = rotate(x, angle)
    x = crop_around_center(x, *largest_rotated_rect(width, width, math.radians(angle)))
    x = cv2.resize(x, dsize=(width, width), interpolation=cv2.INTER_CUBIC)
    return x


def cifar10_confusion_matrix(y_true, y_pred, normalize, out_png):
    title = 'Cifar10 confusion matrix'
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plot_confusion_matrix(y_true, y_pred, class_names, normalize, title, out_png=out_png)
    plt.close()
    return


def food101_confusion_matrix(y_true, y_pred, txt_labels, normalize, out_png):
    title = 'Food101 confusion matrix'
    class_names = txt_labels
    plot_confusion_matrix(y_true, y_pred, class_names, normalize, title, out_png=out_png)
    plt.close()
    return


def cifar100_confusion_matrix(y_true, y_pred, txt_labels, normalize, out_png):
    title = 'CIFAR100 confusion matrix'
    class_names = txt_labels
    plot_confusion_matrix(y_true, y_pred, class_names, normalize, title, out_png=out_png)
    plt.close()
    return


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.get_cmap('Blues'),
                          out_png='confusion_matrix.png'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    #for i in range(cm.shape[0]):
        #for j in range(cm.shape[1]):
            #ax.text(j, i, format(cm[i, j], fmt),
            #        ha="center", va="center",
            #        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(out_png)
    return ax


def create_food101_dataset(dataset_path='/media/benedetta/e257df19-3c9a-4a5e-8ebc-43bc9a6ce05d/Datasets/food-101',
                           target_shape=(299, 299), shuffle=True):

    dataset_shape = f"{target_shape[0]}x{target_shape[1]}"

    os.makedirs(os.path.join(dataset_path, dataset_shape, 'Train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, dataset_shape, 'Test'), exist_ok=True)

    # Classes
    categories_fp = glob.glob(os.path.join(dataset_path, 'images/*/'))
    categories = [os.path.basename(os.path.normpath(x)) for x in categories_fp]

    for t in ['Train', 'Test']:
        for cat in categories:
            os.makedirs(os.path.join(dataset_path, dataset_shape, t, cat), exist_ok=True)

    for cat in tqdm(categories):
        images = glob.glob(os.path.join(dataset_path, 'images', cat, '*.*'))

        if shuffle:
            random.shuffle(images)

        n_images = len(images)
        n_train = int(0.95 * n_images)

        for i, im in enumerate(images):
            img = cv2.imread(im)
            img = cv2.resize(img, target_shape, interpolation=cv2.INTER_CUBIC)

            dst_folder = 'Train' if i < n_train else 'Test'
            file_out = os.path.join(dataset_path, dataset_shape, dst_folder, cat, f"{os.path.splitext(os.path.basename(im))[0]}.png")
            cv2.imwrite(file_out, img)
    return


def load_gtsrdb_data(dataset, data_dir, img_size=32):
    images = []
    classes = []
    rows = pd.read_csv(dataset)
    rows = rows.sample(frac=1).reset_index(drop=True)

    with tqdm(total=len(list(rows.iterrows()))) as pbar:
        for i, row in rows.iterrows():
            img_class = row["ClassId"]
            img_path = row["Path"]
            image = os.path.join(data_dir, img_path)

            image = cv2.imread(image)
            image_rs = cv2.resize(image, (img_size, img_size), 3)
            R, G, B = cv2.split(image_rs)
            img_r = cv2.equalizeHist(R)
            img_g = cv2.equalizeHist(G)
            img_b = cv2.equalizeHist(B)
            new_image = cv2.merge((img_r, img_g, img_b))

            images.append(new_image)
            classes.append(img_class)

            pbar.update(1)

    X = np.array(images)
    y = np.array(classes)

    return X, y


def plot_gtsrdb_history_keys(h):

    plt.plot(h['accuracy'])
    plt.plot(h['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid()
    plt.show()
    plt.savefig('gtsrdb_accuracy.png')
    plt.close()

    # summarize history for loss
    plt.plot(h['loss'])
    plt.plot(h['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid()
    plt.show()
    plt.savefig('gtsrdb_loss.png')
    plt.close()

    return


def gtsrb_subset(trainX, trainY, testX, testY, cutoff=0.70):

    num_labels = len(np.unique(trainY))

    trainX1 = []
    trainY1 = []
    testX1 = []
    testY1 = []
    for y in range(0, num_labels):
        idx = np.array(np.where(trainY == y)).flatten()
        n = len(idx)
        rperm = np.random.permutation(n)
        trainX1.append(trainX[idx][rperm[:int(n * cutoff)]])
        trainY1.append(y * np.ones(int(n * cutoff)))
    trainX1 = np.array(trainX1)
    trainY1 = np.array(trainY1)

    x = np.zeros((1, 32, 32, 3))
    for z in trainX1:
        x = np.concatenate((x, z), axis=0)
    trainX = x[1:]

    x = np.array([-1])
    for z in trainY1:
        x = np.concatenate((x, z), axis=0)
    trainY = x[1:]

    for y in range(0, num_labels):
        idx = np.array(np.where(testY == y)).flatten()
        n = len(idx)
        rperm = np.random.permutation(n)
        testX1.append(testX[idx][rperm[:int(n * cutoff)]])
        testY1.append(y * np.ones(int(n * cutoff)))
    testX1 = np.array(testX1)
    testY1 = np.array(testY1)

    x = np.zeros((1, 32, 32, 3))
    for z in testX1:
        x = np.concatenate((x, z), axis=0)
    testX = x[1:]

    x = np.array([-1])
    for z in testY1:
        x = np.concatenate((x, z), axis=0)
    testY = x[1:]

    return trainX, trainY, testX, testY
