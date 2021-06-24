import cv2 
from skimage.color import rgb2gray
from src.autoencoder import AutoEncoder
import mahotas
import numpy as np 
import os 

"""
    Nguồn tham khảo: https://anmol19005.medium.com/content-based-image-retrieval-without-metadata-560c3a37f1c
"""

def his_feature(image, bins=10):
    """
        ACC: 0.95   0.898   0.8305
    """
    hist = cv2.calcHist([image], [0,1,2], None, [bins, bins, bins], [0,256,0,256,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def hu_moment_feature(image):
    """
        ACC: 0.385
    """
    return cv2.HuMoments(cv2.moments(rgb2gray(image))).flatten()


# winSize = (16,16)
# blockSize = (8,8)
# blockStride = (4,4)
# cellSize = (4,4)
# nbins = 9
# hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
hog = cv2.HOGDescriptor()
def hog_feature(image):
    """
        https://stackoverflow.com/questions/28390614/opencv-hogdescripter-python
        ACC: 0.778
    """
    h = hog.compute(image)
    return h.reshape(-1)

def haralick_feature(image):
    """
        Tốt với các ảnh bề mặt cấu trúc đá hoa văn: https://codelungtung.wordpress.com/2018/07/13/image-descriptor-haralick-texture/
        ACC: 0.442
    """
    return mahotas.features.haralick(image).mean(axis=0)

def convAE_feature(X_train):
    """
        ACC: 0.97   0.921   0.842
    """
    shape_img = X_train[0].shape
    outDir = os.path.join(os.getcwd(), "output", "convAE")
    modelName = "convAE"
    info = {
        "shape_img": shape_img,
        "autoencoderFile": os.path.join(outDir, "{}_autoecoder.h5".format(modelName)),
        "encoderFile": os.path.join(outDir, "{}_encoder.h5".format(modelName)),
        "decoderFile": os.path.join(outDir, "{}_decoder.h5".format(modelName)),
    }
    model = AutoEncoder(modelName, info)
    model.set_arch()
    model.load_models(loss="binary_crossentropy", optimizer="adam")
    # shape_img_resize = shape_img
    # input_shape_model = tuple([int(x) for x in model.encoder.input.shape[1:]])
    output_shape_model = tuple([int(x) for x in model.encoder.output.shape[1:]])
    E_train = model.predict(X_train)
    E_train_flatten = E_train.reshape((-1, np.prod(output_shape_model)))
    return E_train_flatten

def convAE_cifar10_feature(X_train):
    """
        ACC: 
    """
    shape_img = X_train[0].shape
    outDir = os.path.join(os.getcwd(), "output", "convAE_cifar10")
    modelName = "convAE"
    info = {
        "shape_img": shape_img,
        "autoencoderFile": os.path.join(outDir, "{}_autoecoder.h5".format(modelName)),
        "encoderFile": os.path.join(outDir, "{}_encoder.h5".format(modelName)),
        "decoderFile": os.path.join(outDir, "{}_decoder.h5".format(modelName)),
    }
    model = AutoEncoder(modelName, info)
    model.set_arch()
    model.load_models(loss="binary_crossentropy", optimizer="adam")
    # shape_img_resize = shape_img
    # input_shape_model = tuple([int(x) for x in model.encoder.input.shape[1:]])
    output_shape_model = tuple([int(x) for x in model.encoder.output.shape[1:]])
    E_train = model.predict(X_train)
    E_train_flatten = E_train.reshape((-1, np.prod(output_shape_model)))
    return E_train_flatten