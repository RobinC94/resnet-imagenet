import os
import keras

from keras.layers.convolutional import Conv2D
from termcolor import cprint
from keras.utils import multi_gpu_model
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

from model_modify import cluster_model_kernels, modify_model, save_cluster_result, load_cluster_result
from model_train_and_test import evaluate_model, fine_tune_model

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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8
    #set_session(tf.Session(config=config))

    model = ResNet50(include_top=True, weights='imagenet')
    #model = multi_gpu_model(model, 2)
    #model.summary()
    model.load_weights("./weights/resnet50_73_69.h5")

    keras.utils.plot_model(model, to_file="./tmp/resnet50.png")
    print_conv_layer_info(model)
    #evaluate_model(model)

    kmeans_k = 1024
    file = "./tmp/resnet50_" + str(kmeans_k)

    #cluster_id, temp_kernels = cluster_model_kernels(model, k=kmeans_k, t=1)
    #save_cluster_result(cluster_id, temp_kernels, file)
    cluster_id, temp_kernels = load_cluster_result(file)

    model_new = modify_model(model, cluster_id, temp_kernels)

    print "start fine-tuneing"
    #print "start testing"
    evaluate_model(model)
    #fine_tune_model(model, epochs=20)



if __name__ == "__main__":
    main()
