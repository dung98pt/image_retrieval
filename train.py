from src.autoencoder import AutoEncoder
import tensorflow as tf 
import os 
from dataset import load_dataset
# Build models

outDir = os.path.join(os.getcwd(), "output", "convAE")
(X_train, y_train), (X_test, y_test) = load_dataset(is_test=False, normalize=True)
shape_img = X_train[0].shape
modelName = "convAE"
trainModel = True

if modelName in ["simpleAE", "convAE"]:

    # Set up autoencoder
    info = {
        "shape_img": shape_img,
        "autoencoderFile": os.path.join(outDir, "{}_autoecoder.h5".format(modelName)),
        "encoderFile": os.path.join(outDir, "{}_encoder.h5".format(modelName)),
        "decoderFile": os.path.join(outDir, "{}_decoder.h5".format(modelName)),
    }
    model = AutoEncoder(modelName, info)
    model.set_arch()

    if modelName == "simpleAE":
        shape_img_resize = shape_img
        input_shape_model = (model.encoder.input.shape[1],)
        output_shape_model = (model.encoder.output.shape[1],)
        n_epochs = 200
    elif modelName == "convAE":
        shape_img_resize = shape_img
        input_shape_model = tuple([int(x) for x in model.encoder.input.shape[1:]])
        output_shape_model = tuple([int(x) for x in model.encoder.output.shape[1:]])
        n_epochs = 200
    else:
        raise Exception("Invalid modelName!")

# Train (if necessary)
if modelName in ["simpleAE", "convAE"]:
    print("0")
    if trainModel:
        print("1")
        model.compile(loss="binary_crossentropy", optimizer="adam")
        model.fit(X_train, n_epochs=n_epochs, batch_size=256)
        model.save_models()
    else:
        print("2")
        model.load_models(loss="binary_crossentropy", optimizer="adam")