class Layers:

    # 层名称
    layer_name = ""

    # 卷积核大小
    k = 0

    # 输入通道
    c_in = 0

    # 输出通道
    c_out = 0

    # 步长
    s = 0

    # 填充
    p = 0

    def __init__(self, layer_name, k, c_in, c_out, s):
        self.layer_name = layer_name
        self.k = k
        self.c_in = c_in
        self.c_out = c_out
        self.s = s
        self.p = int(k/2)