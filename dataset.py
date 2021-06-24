import os 
import skimage.io
import skimage
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import numpy as np 

# Read image
def read_img(filePath):
    return skimage.io.imread(filePath, as_gray=False)

# Normalize image data [0, 255] -> [0.0, 1.0]
def normalize_img(img):
    return img / 255.

# Resize image
def resize_img(img, shape_resized):
    img_resized = resize(img, shape_resized,
                         anti_aliasing=True,
                         preserve_range=True)
    assert img_resized.shape == shape_resized
    return img_resized

# Flatten image
def flatten_img(img):
    return img.flatten("C")

def load_dataset(is_test=True, normalize=False):
    path = "data/coil-100"
    file_names = [i for i in os.listdir(path) if ".png" in i]
    if is_test:
        total = 72 *2
    else:
        total = len(file_names)

    imgs = []
    labels = []
    for i in file_names[:total]:
        labels.append(i.split("_")[0])
        imgs.append(read_img(os.path.join(path, i)))

    input_shape_model = imgs[0].shape
    # print([labels.count(i) for i in set(labels)])
    print("img shape:", imgs[0].shape)
    if normalize:
        imgs = [normalize_img(i) for i in imgs]
    imgs = np.array(imgs).reshape((-1,) + input_shape_model)
    X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.0138, random_state=42, stratify=labels)
    print("Số lớp: {} - {}".format(len(set(labels)), set(labels)))
    print("Số mẫu train: {}, Số mẫu test: {}".format(len(y_train), len(y_test)))
    return (X_train, y_train), (X_test, y_test)

def load_cifar10_dataset():
    X_train = np.load("data/cifar10_trainx.npy")
    y_train = np.load("data/cifar10_trainy.npy")
    X_test =  np.load("data/cifar10_testx.npy")
    y_test =  np.load("data/cifar10_testy.npy")
    print(X_train.shape)
    return (X_train, y_train), (X_test, y_test)