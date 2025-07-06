from Config.Layer import Layers
from Config.Device import Devices
from keras.models import load_model
import numpy as np
from sys import getsizeof

input_shape = []
data_size = []
L = []


d0 = Devices(26327, 1.4 * 1000 ** 3, 4 * 1024 * 1024, 4, 10, [12.8 * 1000 * 1000,1 * 1000], "192.168.198.133")
d1 = Devices(26327, 3.2 * 1000 ** 3, 4 * 1024 * 1024, 15, 10, [1 * 1000, 12.8 * 1000 * 1000], "192.168.198.132")

D = [d0,d1]

# partition_result = [[0, 60, 64], [0, 7, 64], [0, 7, 64], [0, 7, 64], [0, 3, 32], [0, 3, 32], [0, 2, 16], [0, 2, 16], [0, 2, 16], [0, 2, 16], [0, 1, 8], [0, 1, 8], [0, 1, 8], [0, 1, 8], [0, 0, 4], [0, 0, 4], [0, 0, 4], [0, 0, 4]]
# layer_partition_index = [0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 17]
#
#
# partition_result = [[0, 60, 224], [0, 7, 224], [0, 7, 224], [0, 7, 224], [0, 3, 112], [0, 3, 112], [0, 2, 56], [0, 2, 56], [0, 2, 56], [0, 2, 56], [0, 1, 28], [0, 1, 28], [0, 1, 28], [0, 1, 28], [0, 0, 14], [0, 0, 14], [0, 0, 14], [0, 0, 14]]
# layer_partition_index = [0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 17]

device_index = 0
device_num = len(D)

model = load_model("/home/name1/experiment/eDCC_grpc/model/VGG-16.h5")

warm_up_data = np.random.random((1, 64, 64, 3))
model.predict(warm_up_data)

for layer in model.layers:
    input_shape.append(warm_up_data.shape)
    data_size.append(getsizeof(np.asarray(warm_up_data)))
    # data_size.append(getsizeof(np.asarray(warm_up_data).flatten().astype(float).tobytes()))
    kernel_size = 0
    s = 1
    if "pool" in str(layer.name):
        kernel_size = 2
        s = 2
    for v in layer.variables:
        if v.name == str(layer.name) + '/kernel:0':
            kernel_size = v.shape[0]
    p = int(kernel_size / 2)
    c_in = warm_up_data.shape[3]
    warm_up_data = np.asarray(layer(warm_up_data), dtype=float)
    if "sequential" in str(layer.name):
        c_out = 10
    else:
        c_out = warm_up_data.shape[3]
    l = Layers(layer_name=layer.name, k=kernel_size, c_in=c_in, c_out=c_out, s=s)
    # print(str(layer.name) + " " + str(kernel_size) + " " + str(c_in) + " " + str(c_out) + " " + str(s))
    L.append(l)


def getDevicesInNet():
    return device_index,device_num

def getMyIp():
    my_ip = D[device_index].ipAddr
    return my_ip

def getLayerInfo():
    return L

def getDevicesInfo():
    return D

def getInputShape():
    return input_shape, data_size


# def getPartitionResult():
#     return np.asarray(partition_result), np.asarray(layer_partition_index)
