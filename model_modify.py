import os,sys

import numpy as np
import scipy.stats as stats

from keras.layers.convolutional import Conv2D
from termcolor import cprint
from Bio import Cluster

####################################
##config params
pair_layers_num = 1
r_thresh = 0.95
filter_size = 3
kmeans_k=256

####################################
## public API
def modify_model(model, k=kmeans_k,file_save = None):

    # 1 select conv layers
    conv_layers_list = get_conv_layers_list(model)
    cprint("selected conv layers is:" + str(conv_layers_list), "red")

    # 2 get kernels stack
    kernels_stack = get_kernels_stack(model, conv_layers_list)
    print "num of searched kernels:" + str(len(kernels_stack.keys()))

    modify_kernels_stack(kernels_stack,k=k,f_save=file_save)
    set_modified_kernels_stack_to_model(model, kernels_stack, conv_layers_list)

def load_modified_model_from_file(model, file_load = None):

    # 1 select conv layers
    conv_layers_list = get_conv_layers_list(model)
    cprint("selected conv layers is:" + str(conv_layers_list), "red")

    # 2 get kernels stack
    kernels_stack = get_kernels_stack(model, conv_layers_list)
    print "num of searched kernels:" + str(len(kernels_stack.keys()))

    modify_kernels_from_file(kernels_stack, f_load=file_load)
    set_modified_kernels_stack_to_model(model, kernels_stack, conv_layers_list)

########################################
## private API

def get_conv_layers_list(model):
    '''
        only  choose layers which is conv layer, and its filter_size must be same as param "filter_size"
    '''
    res = []
    layers = model.layers
    for i,l in enumerate(layers):
        if isinstance(l, Conv2D) and l.kernel.shape.as_list()[:2] == [filter_size, filter_size]:
            res += [i]
    return res[:pair_layers_num]

def get_kernels_stack(model, conv_layers_list):
    kernels = []
    index = []

    for l in conv_layers_list:
        weights = model.layers[l].get_weights()[0]  ##0 weights, 1 bias; HWCN
        for i in range(model.layers[l].filters):  ##kernel num
            for s in range(model.layers[l].input_shape[-1]):  # kernel depth
                weights_slice = weights[:, :, s, i]  # HWCK
                kernels += [(weights_slice)]
                index += [(l, i, s)]

    kernels_stack = {key: value for key, value in zip(index, kernels)}
    return kernels_stack


def least_square(dataa, datab):
    assert (dataa.shape == datab.shape)
    dataa = dataa.reshape(-1)
    datab = datab.reshape(-1)
    a, b = np.polyfit(dataa, datab, 1)  ##notice the direction: datab = a*dataa + b
    #err = np.sum(np.abs(a * dataa + b - datab))
    return (a, b)



def modify_kernels_stack(kernels_stack,k=kmeans_k,f_save=None):
    kernels_keys = kernels_stack.keys()
    kernels_num = len(kernels_keys)
    kernels_array = np.zeros((kernels_num, filter_size ** 2))

    for i in range(kernels_num):
        kernel_id = kernels_keys[i]
        kernels_array[i] = kernels_stack[kernel_id].flatten()

    print "start clustering"

    clusterid, cdata, avg_r = cluster_kernels(kernels_array, k, 1)

    print "end clustering"

    for i in range(kernels_num):
        cent_id = clusterid[i]
        kernel = kernels_array[i]
        cent = cdata[cent_id]
        a, b = least_square(cent, kernel)
        kernel_id = kernels_keys[i]
        kernels_stack[kernel_id] = a * cent.reshape(filter_size, filter_size) + b

    print "average r2: %6.4f\t" % (avg_r)

    if f_save != None:
        f_clusterid = f_save + "_clusterid.npy"
        f_cdata = f_save + "_cdata.npy"
        np.save(f_clusterid, clusterid)
        np.save(f_cdata, cdata)

def cluster_kernels(kernels_array, k=kmeans_k,times=5):
    n=np.shape(kernels_array)[0]
    best_r=0
    for i in range(times):
        clusterid, error, nfound = Cluster.kcluster(kernels_array, nclusters=k, dist='a')
        cdata, cmask = Cluster.clustercentroids(kernels_array, clusterid=clusterid, )
        avg_r=0
        for j in range(n):
            cent_id = clusterid[j]
            kernel = kernels_array[j]
            cent = cdata[cent_id]
            r = abs(stats.pearsonr(kernel, cent)[0])
            avg_r += r / n
        if avg_r>best_r:
            best_cluster=clusterid
            best_cdata=cdata
            best_r=avg_r
    return best_cluster,best_cdata,best_r

def modify_kernels_from_file(kernels_stack, f_load=None):
    kernels_keys = kernels_stack.keys()
    kernels_num = len(kernels_keys)
    kernels_array = np.zeros((kernels_num, filter_size ** 2))

    for i in range(kernels_num):
        kernel_id = kernels_keys[i]
        kernels_array[i] = kernels_stack[kernel_id].flatten()

    try:
        f_clusterid=f_load +"_clusterid.npy"
        f_cdata= f_load+"_cdata.npy"
        clusterid=np.load(f_clusterid)
        cdata=np.load(f_cdata)
        print "loading file done"
    except:
        print "cannot open file"
        sys.exit(0)

    avg_r=0
    for i in range(kernels_num):
        cent_id = clusterid[i]
        kernel = kernels_array[i]
        cent = cdata[cent_id]
        a, b= least_square(cent, kernel)
        r=abs(stats.pearsonr(kernel,cent)[0])
        avg_r += r/kernels_num
        kernel_id=kernels_keys[i]
        kernels_stack[kernel_id]=a*cent.reshape(filter_size,filter_size)+b

    print "average r2: %6.4f\t" % (avg_r)

def set_modified_kernels_stack_to_model(model, kernels_stack, conv_layers_list):
    for l in conv_layers_list:
        weights = model.layers[l].get_weights()
        for i in range(model.layers[l].filters):  ##kernel num
            for s in range(model.layers[l].input_shape[-1]):  # kernel depth
                if kernels_stack.has_key((l, i, s)):
                    weights[0][:, :, s, i] = kernels_stack[(l, i, s)]
                else:
                    weights[0][:,:,s,i]=[[0,0,0],[0,0,0],[0,0,0]]
        model.layers[l].set_weights(weights)


#####################################
## for debug
if __name__ == "__main__":
    from resnet50 import ResNet50
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    model = ResNet50(include_top=True, weights='imagenet')

    r_thresh = 0.92
    pair_layers_num = 15
    file_load = "./tmp/resnet50_" + str(r_thresh) + "_" + str(pair_layers_num) + ".txt"

    conv_layers_list = get_conv_layers_list(model)
    cprint("selected conv layers is:" + str(conv_layers_list), "red")

    kernels_stack = get_kernels_stack(model, conv_layers_list)
    print "num of searched kernels:" + str(len(kernels_stack.keys()))

    f_load = open(file_load, "r")
    pair_res = load_pairs_from_file(f_load, kernels_stack)
    print "num of pairs is:" + str(len(pair_res))
    f_load.close()

    modify_kernels_stack(kernels_stack, pair_res)
    set_modified_kernels_stack_to_model(model, kernels_stack, conv_layers_list)

