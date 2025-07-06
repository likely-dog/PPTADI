import sys
import threading
import time
import Config.getInfo
from executionEngine import computePartition
import numpy as np
L = Config.getInfo.getLayerInfo()

device_index, device_nums = Config.getInfo.getDevicesInNet()


class InferenceTask:
    # 设备编号
    device_index = 1
    # 工作负载部分
    workload = np.asarray([])
    # 工作负载在整张图片的上行号（包含）
    ceiling = -1
    # 工作负载在整张图片的下行号（不包含）
    bottom = -1
    # 已经完成的层号
    finished_partition = -2
    finished_layer = -2
    # 中间填充
    padding_ceiling_list = {}
    padding_bottom_list = {}
    start_time = 0

    # 分区数组，一个分区结果的数组，表明了所有设备应该负责那些部分的计算
    partition_result_list = []
    # 分区索引，partition_index_list[i]表明了模型中编号为i的层应该按照partition_result_list[]中的那一个分区结果来执行计算
    partition_index_list = []

    computeTime = 0
    transTime = 0
    pureComputeTime = 0

    def __init__(self, workload, ceiling, bottom, partition_result, partition_index):
        self.workload = workload
        self.ceiling = ceiling
        self.bottom = bottom
        self.partition_result_list = partition_result
        self.partition_index_list = partition_index
        self.start_time = time.time()
        self.finished_layer = -1
        self.finished_partition = -1
        self.computeTime = 0
        self.transTime = 0

    def execute(self):
        start_time = time.time()
        print("=" * 50)
        for index in range(len(self.partition_result_list)):
            print("开始执行第" + str(index) + "个分区")
            computePartition(self, index)
            # self.finished_partition += 1
            print("已完成第" + str(self.finished_partition) + "个分区")
            print("=" * 50)
        print("识别结果为：" + str(np.argmax(self.workload)) + " 总耗时：" + str(time.time() - start_time) + "s")
        print("计算耗时：" + str(self.computeTime) + " 传输耗时：" + str(self.transTime))

    def run(self):
        t = threading.Thread(target=InferenceTask.execute, args=(self,))
        t.start()


