#!/usr/bin/python
import sys, os
from termcolor import cprint
from random import sample
import itertools
import threading

from keras.applications import imagenet_utils
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
# from keras.applications.imagenet_utils import preprocess_input
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard

import numpy as np
import json
import xml.etree.ElementTree

##configuration parameters
alpha = 1.0
img_size = 224
img_parent_dir = "/data1/datasets/imageNet/ILSVRC2016/ILSVRC/Data/CLS-LOC/"  # sub_dir: train, val, test

pair_layers_num = 13
filter_size = 3

nb_epoch = 320
batch_size = 32
evaluating_batch_size = 96

##used in 3rd-party model function
class_parse_file = "./tmp/imagenet_class_index.json"
imagenet_utils.CLASS_INDEX = json.load(open(class_parse_file))
# used internally
debug_flag = False


## public API
def evaluate_model1(model, img_path, top=5):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    img = imagenet_utils.preprocess_input(img)  ##mobilenet uses its own way to preprocess images
    preds = model.predict(img)
    print 'Predicted:', decode_predictions(preds, top)
    return decode_predictions(preds, top)


def evaluate_model2(model):
    nb_eval = 50000
    data_gen = evaluating_data_gen()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
    res = model.evaluate_generator(generator=data_gen, steps=nb_eval / evaluating_batch_size, workers=8)
    cprint("modified acc:" + str(res[1]), "red")


def fine_tune(model):
    # fix paired layers
    conv_layers_list = get_conv_layers_list(model)
    for layer_index in conv_layers_list:
        model.layers[layer_index].trainable = False
    print "#########################"
    # compile model to make modification effect!!!
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=1e-3), metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-10), metrics=['accuracy'])
    # fine tune
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger('./fine_tune_mobilenet_cifar10.csv')
    ckpt = ModelCheckpoint(filepath="./fine_tune_weights.h5", monitor='loss', save_best_only=True,
                           save_weights_only=True)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_images=True)
    model.fit_generator(generator=training_data_gen(),
                        steps_per_epoch=1281167 / batch_size,  # 1281167 is the number of training data we have
                        validation_data=evaluating_data_gen(),
                        validation_steps=50000 / evaluating_batch_size,
                        epochs=nb_epoch, verbose=1, max_q_size=32,
                        workers=8,
                        use_multiprocessing=False,
                        callbacks=[lr_reducer, early_stopper, csv_logger, ckpt])
    cprint("fine tune is done\n", "yellow")


##private API
def training_data_gen():
    datagen = ImageDataGenerator(

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images

        preprocessing_function=imagenet_utils.preprocess_input)

    img_dir = os.path.join(img_parent_dir, "train")
    img_generator = datagen.flow_from_directory(
        directory=img_dir,
        target_size=(img_size, img_size),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True)

    return img_generator


class thread_safe_decorator:
    def __init__(self, generator):
        self.generator = generator
        self.lock = threading.Lock()

    def next(self):
        with self.lock:
            batch_data_list = self.generator.next()
        imgs = get_evaluating_batch_imgs(batch_data_list)
        labels = get_evaluating_image_labels(batch_data_list)
        return imgs, labels


def evaluating_data_gen():
    def img_generator():

        img_parent_dir = "/data1/datasets/imageNet/ILSVRC2016/ILSVRC/Data/CLS-LOC/"
        img_dir = os.path.join(img_parent_dir, "val")
        file_data_list = os.listdir(img_dir)
        file_data_array = [file_data_list[i * evaluating_batch_size:(i + 1) * evaluating_batch_size] for i in
                           range(len(file_data_list) / evaluating_batch_size)]
        if debug_flag:
            print len(file_data_array)
            print file_data_array
        iterator = itertools.cycle(file_data_array)
        while (1):
            batch_data_list = iterator.next()
            yield batch_data_list

    return thread_safe_decorator(img_generator())


def get_evaluating_batch_imgs(batch_data_list):
    img_parent_dir = "/data1/datasets/imageNet/ILSVRC2016/ILSVRC/Data/CLS-LOC/"
    img_dir = os.path.join(img_parent_dir, "val")
    img_array = np.empty(shape=(evaluating_batch_size, img_size, img_size, 3), dtype=float)

    for index, i in enumerate(batch_data_list):
        img_path = os.path.join(img_dir, i)
        if debug_flag:
            print img_path
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img = image.img_to_array(img)
        img = imagenet_utils.preprocess_input(img)  ##mobilenet uses its own way to preprocess images
        img_array[index, ...] = img
    return img_array


def get_evaluating_image_labels(batch_data_list):
    img_parent_dir = "/data1/datasets/imageNet/ILSVRC2016/ILSVRC/Annotations/CLS-LOC/"
    img_dir = os.path.join(img_parent_dir, "val")
    indice_array = np.empty(shape=(evaluating_batch_size, 1000), dtype=float)

    for index, i in enumerate(batch_data_list):
        i = i[:-4] + "xml"
        file_name = os.path.join(img_dir, i)
        digit_name = xml.etree.ElementTree.parse(file_name).getroot().findall("object")[0].findall("name")[0].text
        indice = digit_indice_dict[digit_name]
        indice = to_categorical(indice, num_classes=1000)
        indice_array[index] = indice
    return indice_array


def generate_digit_indice_dict():
    digit_indice_dict = {value[0]: int(key) for key, value in imagenet_utils.CLASS_INDEX.items()}
    return digit_indice_dict


def get_conv_layers_list(model):
    '''
        only  choose layers which is conv layer, and its filter_size must be same as param "filter_size"
    '''
    res = []
    layers = model.layers
    for i, l in enumerate(layers):
        if isinstance(l, Conv2D) and l.kernel.shape.as_list()[:2] == [filter_size, filter_size]:
            res += [i]
    return res[:pair_layers_num]


# private data member
digit_indice_dict = generate_digit_indice_dict()

##for debug:
if __name__ == "__main__":
    debug_flag = False
    test = [5, 6]
    '''
    if 1 in test:
        # 1: check ImageDataGenerator's label is same as official defined,
        # data member "class_indices" contain the mapping info of ImageDataGenerator: digit_name:indice
        # official defined mapping file "imagenet_class_index.json": indice_str:[digit_name, string]
        official_mapping = imagenet_utils.CLASS_INDEX
        official_mapping = {value[0]: int(key) for key, value in official_mapping.iteritems()}
        # print official_mapping
        data_gen = training_data_gen()
        generator_mapping = data_gen.class_indices
        # print generator_mapping
        assert (official_mapping == generator_mapping)
        cprint("generator infered mapping is same as official defined, so ImageDataGenerator can be used", "green")
        data, label = data_gen.next()
        # print label.shape

    if 2 in test:
        # 2 check digit_indice_dict
        print digit_indice_dict

    if 3 in test:
        # 3 check evaluating data generator; note: digit string in image file name isn't equal to img's digit name is evaluation dataset
        data_gen = evaluating_data_gen()
        imgs, labels = data_gen.next()
        indice_list = labels.argmax(axis=1)
        cprint(imgs.shape, "red")
        cprint(labels.shape, "red")
        for i in indice_list:
            print imagenet_utils.CLASS_INDEX[str(i)][0]

    if 4 in test:
        # 4 check get_conv_layers_list
        weights_dir = "./weights"
        model = instanciate_mobilenet(weights_dir)
        print get_conv_layers_list(model)

    if 5 in test:
        # 5 test get_kernel_stack and set function; result before set and after set must be same
        weights_dir = "./weights"
        model = instanciate_mobilenet(weights_dir)
        img_path = "/media/flex/d/guizi/data/imageNet/2016/ILSVRC/Data/CLS-LOC/train/n03884397/n03884397_993.JPEG"
        res1 = evaluate_model1(model, img_path)

        conv_layers_list = get_conv_layers_list(model)
        kernels_stack = get_kernels_stack(model, conv_layers_list)
        set_modified_kernels_stack_to_model(model, kernels_stack, conv_layers_list)
        res2 = evaluate_model1(model, img_path)
        assert (res1 == res2)
        cprint("get kernel slice and set API ok", "green")

    if 6 in test:
        # check fine_tune
        fine_tune(model)
'''
