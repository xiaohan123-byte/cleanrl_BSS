'''
一个把 JSON 字段挂到对象上的壳。
把传入的字段挂到对象上，方便后续代码当作对象属性访问。
'''

class BSSChargingScheduling:
    def __init__(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)








