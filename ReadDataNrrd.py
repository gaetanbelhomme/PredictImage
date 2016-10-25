import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage
import os
from six.moves import cPickle as pickle
import nrrd
import sys

# Folders :
# TestingData : TestingData
# TrainingData : TrainingData

# test = "TestingData"
# train = "TrainingData"

# Get FolderName :

root = os.getcwd()
if np.size(sys.argv) != 5:
    print "Error : It must be : python", sys.argv[0], "TestingFolder TrainingFolder OutputFileName filetype [npz/pickle]"
    exit()

test = sys.argv[1]
train = sys.argv[2]
OutputFile = sys.argv[3]

# test = "TestingData/"
# train = "TrainingData/"

start = "V06"
end = "V24"

# Check the presence of folders
# Add the path to the folders


def CheckFolders(root_name, folder):
    FolderPath = os.path.join(root_name, folder)

    if os.path.isdir(FolderPath):
        print folder, "is present"
        return FolderPath
    else:
        raise Exception('Failed to find %s' % folder)

test = CheckFolders(root,test)
train = CheckFolders(root, train)

test_start = CheckFolders(test, start)
test_end = CheckFolders(test, end)

train_start = CheckFolders(train, start)
train_end = CheckFolders(train, end)


image_size = 64
pixel_depth = 255.0


# Normalise les images
# Return dataset, images array

def load_images(folder):
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)
    print(folder)
    num_images = 0
    for Image in image_files:
        image_file = os.path.join(folder, Image)
        try:
            nrrd_image, _ = nrrd.read(image_file)
            image_data = (nrrd_image.astype(float) - pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images += 1
        except IOError as e:
            print('Could not read: ', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    print ('Full dataset tensor:', dataset.shape)
    print ('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


# Construit un fichier pickle dans lequel sont sauvees les images normalisees.
# Renvoie le nom de ces fichiers pickle.

def maybe_pickle(start_folder, end_folder, force=False):
    dataset_names = []
    for folder in start_folder, end_folder:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_images(folder)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names

train_datasets = maybe_pickle(train_start, train_end)
test_datasets = maybe_pickle(test_start, test_end)


# Build 2 arrays according a number of images:
# - dataset array
# - labels array
def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset_s = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        dataset_e = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    else:
        dataset_s, dataset_e = None, None
    return dataset_s, dataset_e


# Settings : pickle_files (folder sorted by labels with its images), train size & valid size.
# Permet de bien arranger les donnees, cad de sortir un vecteur contenant le nombre d images voulues,
# avec autant d'images par classe, ainsi que le vecteur de label associe.
# Creation d'un vecteur de la taille du nombres d images voulues, pour le train et le valid.
# Recuperation du nombre d'images par classe pour le valid et le train dataset(floor(valid_size, num_classes)
# Ouverture du pickle, cad d'une classe.
# Recuperation des images de la classe et melange de ces images.
# Recuperation de toutes du nombre d'images defini ci-dessuse que l'on met dans un vecteur.
# Creation du vecteur de labels associes. Pareil pour le train et le valid dataset.
# Retour de tous lse vecteurs :
# valid_dataset, valid_labels, train_dataset, train_label


def merge_datasets(pickle_files, train_size, valid_size=0):
    valid_dataset_start, valid_dataset_end = make_arrays(valid_size, image_size)
    validation_sets = [valid_dataset_start, valid_dataset_end]

    train_dataset_start, train_dataset_end = make_arrays(train_size, image_size)
    train_sets = [train_dataset_start, train_dataset_end]

    # # let's shuffle the images to have random validation and training set
    #permutation = np.random.permutation(train_sets[0].shape[0])

    start_v, start_t = 0, 0
    end_v, end_t = valid_size, train_size
    end_l = valid_size + train_size
    for i in 0, 1:
        try:
            with open(pickle_files[i], 'rb') as f:
                data_set = pickle.load(f)
                # # let's shuffle the images to have random validation and training set
                #data_set = data_set[permutation, :, :]

                if validation_sets[i] is not None:
                    valid_data = data_set[:valid_size, :, :]
                    validation_sets[i][start_v:end_v, :, :] = valid_data

                train_data = data_set[valid_size:end_l, :, :]
                train_sets[i][start_t:end_t, :, :] = train_data
        except Exception as e:
            print('Unable to process data from', pickle_files[i], ':', e)
            raise
    return validation_sets[0], validation_sets[1], train_sets[0], train_sets[0]


train_size = 2000
valid_size = 100
test_size = 1000

valid_dataset_start, valid_dataset_end, train_dataset_start, train_dataset_end = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset_start, test_dataset_end = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset_start.shape, train_dataset_end.shape)
print('Validation:', valid_dataset_start.shape, valid_dataset_end.shape)
print('Testing:', test_dataset_start.shape, test_dataset_end.shape)


def randomize(dataset_s, dataset_e):
  permutation = np.random.permutation(dataset_s.shape[0])
  shuffled_dataset_s = dataset_s[permutation, :, :]
  shuffled_dataset_e = dataset_e[permutation, :, :]
  return shuffled_dataset_s, shuffled_dataset_e

train_dataset_start, train_dataset_end = randomize(train_dataset_start, train_dataset_end)
test_dataset_start, test_dataset_end = randomize(test_dataset_start, test_dataset_end)
valid_dataset_start, valid_dataset_end = randomize(valid_dataset_start, valid_dataset_end)

if sys.argv[4] == 'npz':
    numpy_file = OutputFile + '.npz'
    try:
        np.savez(numpy_file, train_dataset_start=train_dataset_start, train_dataset_end=train_dataset_end,
                 valid_dataset_start=valid_dataset_start, valid_dataset_end=valid_dataset_end,
                 test_dataset_start=test_dataset_start, test_dataset_end=test_dataset_end)

    except Exception as e:
        print('Unable to save data to', numpy_file, ':', e)
        raise
    statinfo = os.stat(numpy_file)
    print('Numpy file size:', statinfo.st_size)

else:
    pickle_file = OutputFile + '.pickle'
    try:
      f = open(pickle_file, 'wb')
      save = {
        'train_dataset_start': train_dataset_start,
        'train_dataset_end': train_dataset_end,
        'valid_dataset_start': valid_dataset_start,
        'valid_dataset_end': valid_dataset_end,
        'test_dataset_start': test_dataset_start,
        'test_dataset_end': test_dataset_end,
        }
      pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
      f.close()
    except Exception as e:
      print('Unable to save data to', pickle_file, ':', e)
      raise
    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)

