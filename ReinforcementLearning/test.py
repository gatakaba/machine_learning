import numpy as np


class Crawl():
    def __init__(self):

    # self.ser = serial.Serial()

    def get_state(self):
        return np.random.uniform(-0.5, 0.5, size=2)

    def set_motor(self, u):
        pass


crawl = Crawl()
state = crawl.get_state()

q_value = {}

