import grpc
import sys
sys.path.append("/home/name1/experiment/eDCC_grpc/grpc_utils")
from base_package import data_pb2, data_pb2_grpc
import Config
import numpy as np

D = Config.getInfo.getDevicesInfo()
clients = []
device_index, device_num = Config.getInfo.getDevicesInNet()

def initClients():
    for d in D:
        conn = grpc.insecure_channel(d.ipAddr + ':8080')  # 监听频道
        # 客户端使用Stub类发送请求,参数为频道,为了绑定链接
        client = data_pb2_grpc.CoEdgeServerStub(channel=conn)
        clients.append(client)

initClients()

def sendWorkloadTo(target, partition_result_list, partition_index_list, img):

    ceiling = partition_result_list[0][target]
    bottom = partition_result_list[0][target + 1]
    client = clients[target]
    workload = img[:, ceiling:bottom, :, :]
    wid = workload.shape[2]
    wl = workload.astype(float).tobytes()
    pr = partition_result_list.flatten()
    pi = partition_index_list.flatten()
    request = data_pb2.sendWorkloadRequest(workload=wl, ceiling=ceiling, bottom=bottom, wid=wid, partition_result=pr,
                                           partition_index=pi)
    response = client.allocatedWorkload(request)
    return response.code


"""
    全连接层获取其余设备的特征提取阶段计算结果
"""
def getWorkload(wanted_index, required_layer):

    client = clients[wanted_index]
    # 发送请求
    request = data_pb2.workloadrequest(layer=required_layer)
    response = client.GetWorkload(request)
    wl = np.frombuffer(bytes(response.workload), dtype=float)
    return wl, response.height


def sendPaddingTo(target, partition_index, padding):
    if (target < 0) | (target >= device_num):
        return 400
    client = clients[target]
    seq = 0
    if target < device_index:
        seq = 1

    # 发送请求
    padding = np.asarray(padding).flatten().astype(float).tobytes()
    request = data_pb2.paddingrequest(padding=padding, partition=partition_index, up_or_down=seq)
    response = client.GetPadding(request)
    return
