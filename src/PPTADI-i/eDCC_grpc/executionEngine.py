from keras.models import load_model
import numpy as np
import Config.getInfo
import time

from grpc_utils.client import getWorkload, sendPaddingTo

L = Config.getInfo.getLayerInfo()
D = Config.getInfo.getDevicesInfo()
model = load_model("/home/name1/experiment/eDCC_grpc/model/VGG-16.h5")
device_index, device_nums = Config.getInfo.getDevicesInNet()

"""模型预热"""
warm_up_data = np.random.random((1, 64, 64, 3))
model.predict(warm_up_data)
for layer in model.layers:
    warm_up_data = np.asarray(layer(warm_up_data))

"""
    计算前后设备传输来的数据
"""


def computeP(ee, partition_index):
    # 首先计算这个分区本设备需要计算那些数据
    layer_index = ee.finished_layer + 1
    partition_result_list = ee.partition_result_list
    partition_index_list = ee.partition_index_list
    k = layer_index
    p_up_n = 0
    p_down_n = 0
    ceiling = ee.ceiling
    bottom = ee.bottom

    # 应该负责的计算部分
    ceiling_responsible = partition_result_list[partition_index_list[layer_index]][device_index]
    bottom_responsible = partition_result_list[partition_index_list[layer_index]][device_index + 1]

    # 考虑填充部分
    if "pool" in L[layer_index].layer_name:
        if bottom_responsible % 2 == 1:
            bottom_responsible += 1
            # ee.bottom += 1

    else:
        while ee.partition_index_list[k] == partition_index:
            p_up_n += L[k].p
            p_down_n += L[k].p
            k += 1
            if k >= len(ee.partition_index_list):
                break

        ceiling_responsible -= p_up_n
        bottom_responsible += p_down_n

    # 作差得到上下需要的填充高度
    ceiling_needed = ceiling - ceiling_responsible
    bottom_needed = bottom_responsible - bottom

    if device_index == 0:
        ceiling_needed = 0
    if device_index == device_nums - 1:
        bottom_needed = 0

    if "seq" in L[layer_index].layer_name:
        ceiling_needed = 0
        bottom_needed = 0

    return ceiling_needed, bottom_needed, p_up_n


def getPaddingFrom(target_device, partition_index, p, ee):
    if p <= 0:
        return []
    if (target_device < 0) | (target_device >= device_nums):
        return []
    elif target_device < device_index:
        # 轮询是否有需要的填充
        while partition_index - 1 not in ee.padding_ceiling_list.keys():
            continue
        padding = ee.padding_ceiling_list[partition_index - 1]
    else:
        while partition_index - 1 not in ee.padding_bottom_list.keys():
            continue
        padding = ee.padding_bottom_list[partition_index - 1]
    return padding


"""
    获取填充
"""


def getPadding(partition_index, ee):
    p_ceiling_n, p_bottom_n, p = computeP(ee, partition_index)
    padding_ceiling = []
    padding_bottom = []

    if p_ceiling_n > 0:
        padding_ceiling = getPaddingFrom(device_index - 1, partition_index, p_ceiling_n, ee)
    if p_bottom_n > 0:
        padding_bottom = getPaddingFrom(device_index + 1, partition_index, p_bottom_n, ee)

    return padding_ceiling, padding_bottom, p


def joinData(ee, padding_ceiling, padding_bottom):
    if len(padding_ceiling) == 0:
        pass
    else:
        p = int(len(padding_ceiling) / (ee.workload.shape[0] * ee.workload.shape[2] * ee.workload.shape[3]))
        padding_ceiling = np.reshape(padding_ceiling,
                                     (ee.workload.shape[0], p, ee.workload.shape[2], ee.workload.shape[3]))
        ee.workload = np.concatenate((padding_ceiling, ee.workload), axis=1)

    if len(padding_bottom) == 0:
        pass
    else:
        p = int(len(padding_bottom) / (ee.workload.shape[0] * ee.workload.shape[2] * ee.workload.shape[3]))
        padding_bottom = np.reshape(padding_bottom,
                                    (ee.workload.shape[0], p, ee.workload.shape[2], ee.workload.shape[3]))
        ee.workload = np.concatenate((ee.workload, padding_bottom), axis=1)


def adjustCeilingAndBottom(ee):
    # 调整上下界
    ee.ceiling = ee.partition_result_list[ee.finished_partition + 1][device_index]
    ee.bottom = ee.partition_result_list[ee.finished_partition + 1][device_index + 1]

    if "pool" in L[ee.finished_layer + 1].layer_name:

        if ee.ceiling % 2 == 1:
            ee.ceiling += 1

        if ee.bottom % 2 == 1:
            ee.bottom += 1


"""
    计算分层
"""


def computeLayer(ee, execute_layer_index):
    layer = model.layers[execute_layer_index]
    print("----开始执行第" + str(execute_layer_index) + "层" + str(layer.name))

    # 卷积层
    if "conv" in str(L[execute_layer_index].layer_name):
        ee.workload = layer(ee.workload)
        # ee.workload = np.asarray(workload)[:, int(p):int(-p), :, :]

    # 池化层
    elif "pool" in str(L[execute_layer_index].layer_name):
        # 计算并返回
        if ee.ceiling % 2 != 0:
            ee.workload = ee.workload[:, 1:, :, :]
            ee.ceiling += 1
        ee.workload = np.asarray(layer(ee.workload))
        ee.ceiling /= 2
        ee.bottom /= 2

    # 计算能力最强的设备执行最后的全连接层
    elif "sequential" in str(layer.name):
        start_time = time.time()
        if device_index != 0:
            return
        workload = ee.workload
        for index in range(len(D)):
            wl, h = getWorkload(index, execute_layer_index - 1)
            wl = np.reshape(wl, (ee.workload.shape[0], h, ee.workload.shape[2], ee.workload.shape[3]))
            if index == 0:
                workload = wl
            else:
                workload = np.concatenate((workload, wl), axis=1)
        ee.workload = np.asarray(layer(workload))
    ee.finished_layer += 1


"""发送数据给其它设备"""


def sendPadding(ee):
    sendToPre(ee)
    sendToNext(ee)
    return


def sendToPre(ee):
    # 前一个设备下一分区负责计算的范围下界
    next_partition_bottom_pre = ee.partition_result_list[ee.finished_partition + 1][device_index]

    ceiling = ee.partition_result_list[ee.finished_partition + 1][device_index]

    # 下一个分区需要的填充高度
    p = 0

    if "pool" in L[ee.finished_layer + 1].layer_name:
        if next_partition_bottom_pre % 2 != 0:
            p += 1
            ceiling += 1

    else:
        k = ee.finished_layer + 1
        while ee.partition_index_list[k] == ee.finished_partition + 1:
            p += L[k].p
            k += 1
            if k >= len(ee.partition_index_list):
                break
        ceiling -= p

    next_partition_bottom_pre += p

    if int(next_partition_bottom_pre) <= int(ee.ceiling):
        return

    w = int(next_partition_bottom_pre - ee.ceiling)

    padding = ee.workload[:, :w, :, :]

    print("发送给前一个设备，形状为：" + str(padding.shape))
    sendPaddingTo(device_index - 1, ee.finished_partition, padding)

    if ceiling > ee.ceiling:
        ee.workload = ee.workload[:, ceiling - ee.ceiling:, :, :]


def sendToNext(ee):
    # 下一个设备下一分区负责计算的范围上界
    next_partition_ceiling_next = ee.partition_result_list[ee.finished_partition + 1][device_index + 1]

    bottom = ee.partition_result_list[ee.finished_partition + 1][device_index + 1]

    # 下一个分区需要的填充高度
    p = 0
    if "pool" not in L[ee.finished_layer + 1].layer_name:
        k = ee.finished_layer + 1
        while ee.partition_index_list[k] == ee.finished_partition + 1:
            p += L[k].p
            k += 1
            if k >= len(ee.partition_index_list):
                break

        bottom += p

    next_partition_ceiling_next -= p

    if next_partition_ceiling_next >= ee.bottom:
        return

    w = int(ee.bottom - next_partition_ceiling_next)

    padding = ee.workload[:, -w:, :, :]
    print("发送给下一个设备，形状为：" + str(padding.shape))
    sendPaddingTo(device_index + 1, ee.finished_partition, padding)

    if bottom < ee.bottom:
        ee.workload = ee.workload[:, :-(ee.bottom - bottom), :, :]


def computePartition(ee, partition_index):
    """先依据需求获取其他设备发送来的填充"""
    padding_ceiling, padding_bottom, p = getPadding(partition_index, ee)

    """拼接数据"""
    joinData(ee, padding_ceiling, padding_bottom)

    """调整上下界"""
    adjustCeilingAndBottom(ee)

    execute_layer_index = ee.finished_layer + 1
    while (ee.partition_index_list[execute_layer_index] == partition_index):
        start_time = time.time()
        computeLayer(ee, execute_layer_index)
        ee.computeTime += time.time() - start_time
        execute_layer_index += 1
        # 如果当前计算层大于模型层数，则最后一层已经执行完毕
        if execute_layer_index >= len(L):
            return

    ee.workload = ee.workload[:, int(p):int(ee.bottom - ee.ceiling + p), :, :]
    ee.finished_partition += 1
    # 传输数据
    start_time = time.time()
    sendPadding(ee)
    ee.transTime += time.time() - start_time

