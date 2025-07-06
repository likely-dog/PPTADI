import grpc
import time
from concurrent import futures
import numpy as np
import sys
from sys import getsizeof

sys.path.append("/home/name1/experiment/eDCC_grpc")
import Config.getInfo
from InferenceTask import InferenceTask

from base_package import data_pb2, data_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
# _HOST = 'localhost'
_HOST = Config.getInfo.getMyIp()
_PORT = '8080'
ee = InferenceTask(np.asarray([[[[]]]]), -1, -1, [], [])
device_index, device_num = Config.getInfo.getDevicesInNet()


class allocatedWorkload(data_pb2_grpc.CoEdgeServerServicer):
    """接收工作负载"""

    def allocatedWorkload(self, request, context):
        print("=" * 50)
        height = request.bottom - request.ceiling
        wid = request.wid
        workload = np.frombuffer(request.workload, dtype=float)
        if len(workload) == 0:
            return data_pb2.sendWorkloadResponse(code=201)
        workload = np.reshape(workload, (1, height, wid, 3))
        pr = np.asarray(request.partition_result)

        rows = len(pr) // (device_num + 1)  # 使用整除，确保没有余数
        pr = pr.reshape(rows, device_num + 1)
        # pr = pr.reshape(int(len(pr) / (device_num + 1)), device_num + 1)
        pi = np.asarray(request.partition_index)
        global ee
        ee = InferenceTask(workload, request.ceiling, request.bottom, pr, pi)
        ee.run()
        return data_pb2.sendWorkloadResponse(code=200)

    def GetPadding(self, request, context):
        partition = request.partition
        up_or_down = request.up_or_down
        p = request.p
        global ee
        padding = np.frombuffer(request.padding, dtype=float)
        if up_or_down == 0:
            ee.padding_ceiling_list[partition] = padding
        else:
            ee.padding_bottom_list[partition] = padding
        return data_pb2.paddingresponse(wait_time = 0)

    def GetWorkload(self, request, context):
        required_layer = request.layer
        global ee
        while ee.finished_layer != required_layer:
            continue
        workload = np.asarray(ee.workload)
        h = workload.shape[1]
        workload = workload.astype(float).flatten().tobytes()
        return data_pb2.workloadresponse(workload=workload, height=h)


def serve():
    # 定义服务器并设置最大连接数,corcurrent.futures是一个并发库，类似于线程池的概念
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=10))  # 创建一个服务器
    # 在服务器中添加派生的接口服务（自己实现了处理函数
    data_pb2_grpc.add_CoEdgeServerServicer_to_server(allocatedWorkload(), grpcServer)
    grpcServer.add_insecure_port(_HOST + ':' + _PORT)  # 添加监听端口
    grpcServer.start()  # 启动服务器
    print("grpc server started!")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        grpcServer.stop(0)  # 关闭服务器


if __name__ == '__main__':
    serve()
