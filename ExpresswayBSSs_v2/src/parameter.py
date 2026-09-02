'''
业务参数
'''
from dataclasses import dataclass

@dataclass
class parameter:
    SBattery_Volume: int = 70
    '''电池容量，单位 kWh'''
    LBattery_Volume: int = 100
    '''电池容量，单位 kWh'''
    SOCmax: float = 0.95
    SOCinit_min: float = 0.1
    SOCinit_max: float = 0.3
    Power: int = 30

    # Init_Battery: int = 0
    # unit_peak_shaving_price: float = 0.5
    # unit_customer_loss_price: float = 0