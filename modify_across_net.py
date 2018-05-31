import os

import numpy as np
import scipy.stats as stats

from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense
from termcolor import cprint
from array import array

from renset50_modified import ResNet50_Modified
from Bio.Cluster import distancematrix

####################################
##config params
kmeans_k=256
filter_size = 3
d_thresh = 0

####################################
## public API
def modify_across_net(model, temp_kernels, k):
    conv_layers_list = get_conv_layers_list(model)
    cprint("selected conv layers is:" + str(conv_layers_list), "red")

    kernels_array = get_kernels_array(model, conv_layers_list)

    #cluster_id= get_clusterid(kernels_array, temp_kernels)
    print 'link done'

    #np.save("tmp/resnet50_vgg19_" + str(k) + "_clusterid.npy",cluster_id)

    cluster_id=np.load("tmp/resnet50_vgg19_" + str(k) + "_clusterid.npy")

    coef_a, coef_b = get_coefficients(kernels_array, cluster_id, temp_kernels)

    model_new = ResNet50_Modified(include_top=True,
                                  temp=temp_kernels,
                                  clusterid=cluster_id)
    print 'rebuilding model done'


    # 5 set new model weights
    set_cluster_weights_to_model(model, model_new,
                                 coef_a=coef_a,
                                 coef_b=coef_b,
                                 clusterid=cluster_id,
                                 cdata=temp_kernels,
                                 conv_layers_list=conv_layers_list)

    set_cluster_weights_to_old_model(model, cluster_id, temp_kernels, coef_a, coef_b, conv_layers_list)

    return model_new


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
            res+= [i]
    return res

def get_weighted_layers_list(model):
    '''
        get all layers with weights without conv3x3
    '''
    res = []
    layers = model.layers
    for i,l in enumerate(layers):
        if (isinstance(l, Conv2D) and l.kernel.shape.as_list()[:2] != [filter_size, filter_size]) \
                or isinstance(l, BatchNormalization) or isinstance(l, Dense):
            res += [i]
    return res

def get_kernels_array(model, conv_layers_list):
    kernels_num = 0

    kernels_buf=array('d')
    for l in conv_layers_list:
        weights = model.layers[l].get_weights()[0]  ##0 weights, 1 bias; HWCN
        for i in range(model.layers[l].filters):  ##kernel num
            for s in range(model.layers[l].input_shape[-1]):  # kernel depth
                weights_slice = weights[:, :, s, i]  # HWCK
                for w in weights_slice.flatten():
                    kernels_buf.append(w)
                kernels_num+=1

    kernels_array = np.frombuffer(kernels_buf,dtype=np.float).reshape(kernels_num,filter_size**2)
    return kernels_array

def least_square(datax, datay):
    assert (datax.shape == datay.shape)
    datax = datax.reshape(-1)
    datay = datay.reshape(-1)
    a, b = np.polyfit(datax, datay, 1)  ##notice the direction: datay = a*datax + b
    return (a, b)

def get_clusterid(kernels_array, cdata):
    kernels_num = np.shape(kernels_array)[0]
    temp_num = np.shape(cdata)[0]
    clusterid = np.zeros(kernels_num)

    avg_sum = 0
    for i in range(kernels_num):
        kernel = kernels_array[i]
        best_id=0
        best_r=1000
        for j in range(temp_num):
            temp=cdata[j]
            data = np.concatenate(([temp],[kernel]), axis=0)
            matrix = distancematrix(data, dist='a')
            r = matrix[1][0]
            if r < best_r:
                best_r = r
                best_id = j

        clusterid[i] = int(best_id)
        avg_sum += best_r

        if i%10000 == 0:
            print i
    avg = avg_sum / kernels_num

    print "average r2:%6.4f" % (avg)

    return clusterid

def get_coefficients(kernels_array, clusterid, cdata):
    kernels_num = np.shape(kernels_array)[0]
    coef_a = np.zeros(kernels_num)
    coef_b = np.zeros(kernels_num)

    avg_sum = 0
    for i in range(kernels_num):
        cent_id = int(clusterid[i])
        kernel = kernels_array[i]
        cent = cdata[cent_id]
        a, b = least_square(cent, kernel)
        coef_a[i] = a
        coef_b[i] = b
        r = abs(stats.pearsonr(kernel, cent)[0])
        avg_sum += r
    avg = avg_sum / kernels_num

    print "average r2:%6.4f" % (avg)

    return coef_a, coef_b


def set_cluster_weights_to_model(model, model_new, clusterid, cdata, coef_a, coef_b,conv_layers_list):
    kernel_id = 0
    for l in conv_layers_list:
        weights = model.layers[l].get_weights()
        weights_new = model_new.layers[l].get_weights()
        weights_new[2] = weights[1]
        for i in range(model.layers[l].filters):  ##kernel num
            for s in range(model.layers[l].input_shape[-1]):  # kernel depth
                weights_new[0][s,i] = coef_a[kernel_id]
                weights_new[1][s,i] = coef_b[kernel_id]

                cent_id = int(clusterid[kernel_id])
                cent = cdata[cent_id]
                weights_new[3][:,:,s,i]=np.array(cent).reshape(filter_size, filter_size)  # HWCK
                kernel_id += 1

        model_new.layers[l].set_weights(weights_new)

    weighted_layers_list = get_weighted_layers_list(model)
    for l in weighted_layers_list:
        weights = model.layers[l].get_weights()
        model_new.layers[l].set_weights(weights)

def set_cluster_weights_to_old_model(model, clusterid, cdata, coef_a, coef_b, conv_layers_list):
    kernels_id = 0
    for l in conv_layers_list:
        weights = model.layers[l].get_weights()
        for i in range(model.layers[l].filters):  ##kernel num
            for s in range(model.layers[l].input_shape[-1]):  # kernel depth
                cent=int(clusterid[kernels_id])
                temp=cdata[cent]
                a=coef_a[kernels_id]
                b=coef_b[kernels_id]
                weights[0][:, :, s, i] = np.array(a*temp+b).reshape(filter_size, filter_size)
                kernels_id += 1
        model.layers[l].set_weights(weights)


#####################################
## for debug
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ''