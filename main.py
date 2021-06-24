import numpy as np
from numpy.lib.npyio import load 
from dataset import load_dataset, load_cifar10_dataset
from extract_feature import his_feature, hu_moment_feature, hog_feature, haralick_feature, convAE_feature
import os 
import time 
from sklearn.neighbors import NearestNeighbors
from src.CV_plot_utils import plot_query_retrieval, plot_tsne, plot_reconstructions

feature = "his" # try: hog, hu_moment, his, haralick, convAE
# Load dataset
if feature == "convAE":
    (X_train, y_train), (X_test, y_test) = load_dataset(is_test=False, normalize=True)
    # extract feature
    E_train = convAE_feature(X_train)
    E_test = convAE_feature(X_test)
else:
    (X_train, y_train), (X_test, y_test) = load_dataset(is_test=False, normalize=False)
    # extract feature
    if feature=="hog":
        E_train = [hog_feature(i) for i in X_train]
        E_test = [hog_feature(i) for i in X_test]
    elif feature=="hu_moment":
        E_train = [hu_moment_feature(i) for i in X_train]
        E_test = [hu_moment_feature(i) for i in X_test]
    elif feature=="his":
        E_train = [his_feature(i) for i in X_train]
        E_test = [his_feature(i) for i in X_test]
    elif feature=="haralick":
        E_train = [haralick_feature(i) for i in X_train]
        E_test = [haralick_feature(i) for i in X_test]
E_shape = E_train[0].shape
E_train = np.array(E_train).reshape((-1,) +  E_shape)
E_test = np.array(E_test).reshape((-1,) +  E_shape)
print(E_train.shape)

# Fit kNN model on training images
times = []
accs = []
# for n_neighbor in [1,5, 10, 15]:
print("Fitting k-nearest-neighbour model on training images...")
knn = NearestNeighbors(n_neighbors=5, metric="cosine")
knn.fit(E_train)

# Perform image retrieval on test images
print("Performing image retrieval on test images...")
outDir = os.path.join(os.getcwd(), "output", "Test")
acc = []

a = time.time()
for i, emb_flatten in enumerate(E_test):
    _, indices = knn.kneighbors([emb_flatten]) # find k nearest train neighbours
    img_query = X_test[i] # query image
    label_query = y_test[i]
    imgs_retrieval = []
    for idx in indices.flatten():
        imgs_retrieval.append(X_train[idx])
        if y_train[idx]==label_query:
            acc.append(1)
        else:
            acc.append(0)
    outFile = os.path.join(outDir, "{}_retrieval_{}.png".format("Test", i))
    plot_query_retrieval(img_query, imgs_retrieval, outFile)
    if i>2: break
# times.append(time.time()-a)
# accs.append(np.mean(acc))
# print("ACC:",  accs)
# print("time:",  times)