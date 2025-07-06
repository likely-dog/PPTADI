import cvxpy as cp
import numpy as np
import Config.getInfo
import cv2

from threading import Thread

from grpc_utils.client import sendWorkloadTo

"""
    模型相关信息
"""
L = Config.getInfo.getLayerInfo()
l = len(L)
"""
    模拟设备相关信息
"""
D = Config.getInfo.getDevicesInfo()
n = len(D)
"""
    模拟设备能力矩阵
"""
# 设备的计算能力矩阵
# 只有对角线上有元素
# D[i][i]代表设备i执行计算的单位时间
devices_compute_ability = np.zeros((n, n))
for i in range(n):
    devices_compute_ability[i][i] = D[i].getComputeTime()
devices_compute_ability = np.array(devices_compute_ability).reshape(n, n)
# print(devices_compute_ability)
# 主设备的传输能力矩阵
# 只有对角线上有元素
# D[i][i]代表主设备传输给设备i的单位传输时间
main_devices_trans_ability = np.zeros((n, n))
for i in range(n):
    main_devices_trans_ability[i][i] = D[0].getTransportTime(i)
main_devices_trans_ability = np.array(main_devices_trans_ability).reshape(n, n)
# print(main_devices_trans_ability)
# 每个设备的内部传输能力矩阵
# 只有对角线上有元素
# D[i][i]代表设备i内部传输的单位传输时间
devices_trans_self_ability = np.zeros((n, n))
for i in range(n):
    devices_trans_self_ability[i][i] = D[i].getTransportTime(i)
devices_trans_self_ability = np.array(devices_trans_self_ability).reshape(n, n)
# print(devices_trans_self_ability)
# 每个设备的传输能力矩阵
# D[i][j]代表设备i传输给设备j的单位传输时间
devices_trans_ability = []
for i in range(n):
    temp = []
    for j in range(n):
        temp.append(D[i].getTransportTime(j))
    devices_trans_ability.append(temp)
devices_trans_ability = np.array(devices_trans_ability).reshape(n, n)
print(devices_trans_ability)
"""
    图像信息
"""

imgPath = "/home/name1/experiment/3.png"
img = cv2.imread(imgPath)
if img is None:
    print("图像为空！！！！！")
    exit(0)
img = cv2.resize(img, (64, 64))


# 高度
H = img.shape[0]
print("\ntuxianggaodu::{H}\n")

# 一行像素的大小
# wid = imgSize / (H * 1024)
wid = [152, 49304, 1048728, 1048728, 262296, 524440, 524440, 131224, 262296, 262296, 262296, 65688, 131224, 131224,
       131224, 32920, 32920, 32920, 32920, 8344]
wid = np.array(wid) / 1024
# 服务ddl 单位为秒

ddl = 10000

def single_layer_partition(layer):
    result = []
    input_shape, data_size = Config.getInfo.getInputShape()
    input_shape = input_shape[layer]
    H = input_shape[1]
    data_size = data_size[layer]
    """
        Define and solve the CVXPY problem.
        x是列向量
    """
    x = cp.Variable((n, 1))
    """
        基础约束
    """
    constrains_base = [x >= 0, sum(x) == H]

    """传输"""
    # 传输给其它设备的数据矩阵
    trans_mat = np.zeros((n, n))
    for j in range(n):
        if j - 1 >= 0:
            trans_mat[j][j - 1] = L[i].p / H * wid[i]
        if j + 1 < n:
            trans_mat[j][j + 1] = L[i].p / H * wid[i]
    # 传输时间
    time_for_transport_layer_mat = devices_trans_ability @ trans_mat.T
    time_for_transport_layer = []
    for j in range(n):
        time_for_transport_layer.append([time_for_transport_layer_mat[j][j]])
    np.array(time_for_transport_layer).reshape(n, 1)

    """计算"""
    # 执行计算时间
    time_for_compute_layer = (devices_compute_ability @ (x * wid[i]))

    """总时延"""
    time_for_a_layer = time_for_transport_layer + time_for_compute_layer

    max_time = cp.max(time_for_a_layer)

    constrains_time = [max_time <= ddl]

    cons = constrains_base + constrains_time
    # prob = cp.Problem(cp.Minimize(EC + EX), cons)
    prob = cp.Problem(cp.Minimize(max_time), cons)
    prob.solve(solver=cp.CPLEX)
    if prob.status == "infeasible":
        print("No Solution")
        return

    proportion = 0
    # 每部分大小
    proportion_rows = []
    # 每部分行号
    partition = [0]
    for xx in x.value:
        proportion += float(xx)
        proportion_rows.append(round(proportion) - sum(partition))
        partition.append(round(proportion))
    return max_time.value, partition


def multi_layer_partition(begin_layer, end_layer, p):
    result = []
    input_shape, data_size = Config.getInfo.getInputShape()
    """
        Define and solve the CVXPY problem.
        x是列向量
    """
    x = cp.Variable((n, 1))
    """
        基础约束
    """
    constrains_base = [x >= 0, sum(x) == H]

    """传输"""
    # 传输给其它设备的数据矩阵
    trans_mat = np.zeros((n, n))
    for j in range(n):
        if j - 1 >= 0:
            trans_mat[j][j - 1] = p / H * wid[i]
        if j + 1 < n:
            trans_mat[j][j + 1] = p / H * wid[i]
    # 传输时间
    time_for_transport_layer_mat = devices_trans_ability @ trans_mat.T
    time_for_transport_layer = []
    for j in range(n):
        time_for_transport_layer.append([time_for_transport_layer_mat[j][j]])
    np.array(time_for_transport_layer).reshape(n, 1)

    """总时延"""
    time_for_a_layer = time_for_transport_layer
    # 执行计算时间
    for j in range(begin_layer, end_layer):
        time_for_a_layer += (devices_compute_ability @ (x * wid[i]))

    max_time = cp.max(time_for_a_layer)

    constrains_time = [max_time <= ddl]

    cons = constrains_base + constrains_time
    # prob = cp.Problem(cp.Minimize(EC + EX), cons)
    prob = cp.Problem(cp.Minimize(max_time), cons)
    prob.solve(solver=cp.CPLEX)
    if prob.status == "infeasible":
        print("No Solution")
        return
    proportion = 0
    # 每部分大小
    proportion_rows = []
    # 每部分行号
    partition = [0]
    for xx in x.value:
        proportion += float(xx)
        proportion_rows.append(round(proportion) - sum(partition))
        partition.append(round(proportion))
    return max_time.value, partition


def eDCC_partition():
    # 一行像素的大小
    # data_size = imgSize / (H * 1024)
    input_shape, data_size = Config.getInfo.getInputShape()
    data_size = np.array(data_size) / 1024

    # 初始化时间和能耗
    execute_time = 0

    # 上一个分区的开始层号
    pre_partition_begin_index = -1
    # 上一个分区的执行时间
    pre_partition_time = 0
    # 分区结果
    partition_result = []
    # 每一层使用哪一个分区结果
    layer_partition_index = []

    """
        顺序逐层考虑
    """
    for i in range(len(L)):

        """
            第一次分配时间和功耗
        """
        if i == 0:
            x = cp.Variable((n, 1))
            """传输"""
            # temp_mat是一个行向量，第一个值为1，其余为0
            temp_mat = np.zeros((1, n))
            temp_mat[0][0] = 1
            # 列向量乘以行向量
            # trans_mat是一个n行n列的向量
            # 对角线上的元素[n][n]代表设备0传输给设备n的数据量
            trans_mat = x * data_size[0] @ temp_mat
            # 主设备分配给各个工作设备的时间
            # 单位传输时间矩阵乘数据量
            # time_for_origin_transport矩阵对角线上的元素是主设备传输给设备n的时间
            time_for_origin_transport = main_devices_trans_ability @ trans_mat
            # 转换为n行1列
            # time_for_origin_transport是n行1列的矩阵，第i行代表传给设备i所需要的时间
            time_for_origin_transport = time_for_origin_transport @ np.ones((n, 1))

            """计算"""
            # 执行计算时间
            time_for_compute_layer = (devices_compute_ability @ (x * data_size[i]))

            """总时延"""
            time_for_a_layer = time_for_origin_transport + time_for_compute_layer
            max_time = cp.max(time_for_a_layer)

            constrains_base = [x >= 0, sum(x) == 1]

            cons = constrains_base
            # prob = cp.Problem(cp.Minimize(EC + EX), cons)
            prob = cp.Problem(cp.Minimize(max_time), cons)
            prob.solve(solver=cp.CPLEX)
            if prob.status == "infeasible":
                print("No Solution")
                return

            proportion = 0
            # 每部分大小
            proportion_rows = []
            # 每部分行号
            partition = [0]
            for xx in x.value:
                proportion += float(xx * input_shape[i][1])
                proportion_rows.append(round(proportion) - sum(partition))
                partition.append(round(proportion))

            pre_partition_begin_index = 0
            pre_partition_time = max_time.value
            partition_result.append(partition)
            layer_partition_index.append(0)

        else:

            if "pool" in L[i].layer_name:
                temp_time, temp_partition = single_layer_partition(i)
                pre_partition_begin_index = i
                pre_partition_time = temp_time
                partition_result.append(temp_partition)
                layer_partition_index.append(len(partition_result) - 1)

            elif (data_size[i] / input_shape[i][1]) <= (
                    data_size[pre_partition_begin_index] / input_shape[pre_partition_begin_index][1]):
                temp_time, temp_partition = single_layer_partition(i)
                pre_partition_begin_index = i
                pre_partition_time = temp_time
                partition_result.append(temp_partition)
                layer_partition_index.append(len(partition_result) - 1)

            else:
                # 不融合
                temp_time, temp_partition = single_layer_partition(i)

                # 融合
                temp_time_fussed, temp_partition_fussed = multi_layer_partition(pre_partition_begin_index, i,
                                                                                i - pre_partition_begin_index + 1)

                # 如果融合效果不好
                if temp_time_fussed > temp_time + pre_partition_time:
                    pre_partition_begin_index = i
                    pre_partition_time = temp_time
                    partition_result.append(temp_partition)
                    layer_partition_index.append(len(partition_result) - 1)

                else:
                    pre_partition_time = temp_time_fussed
                    layer_partition_index.append(len(partition_result) - 1)

    print("CENGHUAFEN"+str(partition_result)+"\n")
    print("DEVICEHUAFEN"+str(layer_partition_index)+"\n")
    return np.asarray(partition_result), np.asarray(layer_partition_index)

def send_workload():
    imgPath = "/home/name1/experiment/3.png"
    img = cv2.imread(imgPath)
    img = cv2.resize(img, (64, 64))
    img = img.reshape(1, img.shape[0], img.shape[1], 3)

    partition_result, partition_index = eDCC_partition()
    p = partition_result[0]
    device_index, device_num = Config.getInfo.getDevicesInNet()
    D = Config.getInfo.getDevicesInfo()
    a, b = Config.getInfo.getInputShape()

    for i in range(device_num):

        t = Thread(target=sendWorkloadTo, args=(i, partition_result, partition_index, img))
        t.start()

        # sendWorkloadTo(i, p, partition_result, partition_index, img)

if __name__ == '__main__':
    partition_result, partition_index = eDCC_partition()

    data_size = Config.getInfo.data_size
    print("\nDATASIZE"+str(data_size)+"\n")

    k = -1
    for i in range(len(L)):
        if partition_index[i] > k:
            print("="*10)
            print("第" + str(k+1) + "个分区：")
            print(str(partition_result[k+1]))
            k += 1

        print("第" + str(i) + "层：" + str(L[i].layer_name))
        
       
    send_workload()

