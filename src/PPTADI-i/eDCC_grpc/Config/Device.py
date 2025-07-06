class Devices:

    # 计算强度 1KB数据的计算周期数
    compInten = 0
    # 时钟频率 单位为hz
    freq = 0
    # 内存 单位为KB
    m = 0
    # 计算功率
    Pc = 0
    # 传输功率
    Px = 0
    # 带宽 单位为KB/s
    b = []
    # ip
    ipAddr = ""

    def __init__(self, compInten, freq, m, Pc, Px, b, ip):
        self.compInten = compInten
        self.freq = freq
        self.m = m
        self.Pc = Pc
        self.Px = Px
        self.b = b
        self.ipAddr = ip

    def getCompute(self, r):
        # 计算功耗
        T_li_compu = (self.compInten * r) / self.freq
        E_li_compu = self.Pc * T_li_compu
        # print(T_li_compu)
        return T_li_compu, E_li_compu

    def getComputeTime(self):
        # 计算时间 s/KB
        return self.compInten / self.freq

    def getTransport(self, p, i):
        # 传输功耗
        T_li_trans = p / self.b[i]
        E_li_trans = self.Px * T_li_trans
        return T_li_trans, E_li_trans

    def getTransportTime(self, i):
        # 传输功耗
        return 1 / self.b[i]