import os
import keras

import model_modify as modi
import model_train_and_test as trte

from keras.layers.convolutional import Conv2D
from termcolor import cprint

from resnet50 import ResNet50

def print_conv_layer_info(model):
    f = open("./tmp/conv_layers_info.txt", "w")
    f.write("layer index   filter number   filter shape(HWCK)\n")

    cprint("conv layer information:", "red")
    for i, l in enumerate(model.layers):
        if isinstance(l, Conv2D):
            print i, l.filters, l.kernel.shape.as_list()
            print >> f, i, l.filters, l.kernel.shape.as_list()
    f.close()

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    model = ResNet50(include_top=True, weights='imagenet')
    #model.summary()

    modi.pair_layers_num = 16
    trte.pair_layers_num=modi.pair_layers_num
    kmeans_k=1024

    keras.utils.plot_model(model, to_file="./tmp/resnet50.png")
    print_conv_layer_info(model)

    #img_path = "/data1/datasets/imageNet/ILSVRC2016/ILSVRC/Data/CLS-LOC/train/n03884397/n03884397_993.JPEG"
    #trte.evaluate_model1(model, img_path)


    file = "./tmp/resnet50_" + str(kmeans_k) + "_" + str(modi.pair_layers_num)
    modi.modify_model(model, k=kmeans_k,file_save = file)
    #modi.load_modified_model_from_file(model, file_load=file)
    #trte.fine_tune(model)

    trte.evaluate_model2(model)


if __name__ == "__main__":
    main()