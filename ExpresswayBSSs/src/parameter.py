'''
现实环境所需参数
'''

class parameter:
    def __init__(self):
        self.Battery_Volume=70
        # self.charging_efficiency=0.9
        self.SOCmax=0.95
        self.SOCinit_min=0.1
        self.SOCinit_max=0.3
        self.Power=30
        self.TOU=0
        low=0.3874 # 电价范围
        middle=0.7005
        high=1.0761
        self.TOU=[] # 分时电价列表
        for i in range(6):
            self.TOU.append(low)
        for i in range(6,8):
            self.TOU.append(middle)
        for i in range(8,11):
            self.TOU.append(high)
        for i in range(11,18):
            self.TOU.append(middle)
        for i in range(18,21):
            self.TOU.append(high)
        for i in range(21,22):
            self.TOU.append(middle)
        for i in range(22,24):
            self.TOU.append(low)
        
        # 下面是意义不明参数，我可以不管，毕竟是自定义环境
        self.Init_Battery=0
        self.unit_peak_shaving_price=0.5
        self.unit_customer_loss_price=0