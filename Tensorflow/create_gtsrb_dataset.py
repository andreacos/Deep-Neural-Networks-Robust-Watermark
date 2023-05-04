import os
import argparse
import utils
import _pickle as pickle

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create Train.p and Test.p from Kaggle''s GTSRB dataset (https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?resource=download).')
    parser.add_argument('gtsrb_dir',  type=str, help='The (unzipped) downloaded directory)')
    args = parser.parse_args()
    gtsrb_dir = args.gtsrb_dir

    os.makedirs('datasets/GTSRB', exist_ok=True)

    train_data = os.path.join(gtsrb_dir, 'Train.csv')
    test_data = os.path.join(gtsrb_dir, 'Test.csv')

    (trainX, trainY) = utils.load_gtsrdb_data(train_data, gtsrb_dir)
    (testX, testY) = utils.load_gtsrdb_data(test_data, gtsrb_dir)

    with open('datasets/GTSRB/Train.p', 'wb') as file:
        pickle.dump((trainX, trainY), file)

    with open('datasets/GTSRB/Test.p', 'wb') as file:
        pickle.dump((testX, testY), file)