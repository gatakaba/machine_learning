# coding:utf-8
"""
Q-learning習作

- ε-greedy policy

"""

import numpy as np
import serial


class Crawl(object):
    def __init__(self):
        self.ser = serial.Serial()
        self.alpha = 0.9  # 学習率
        self.gamma = 0.5  # 割引率
        self.epsiron = 10 ** -2  # ランダム行動する確率
        self.state_split_num = 10  # 状態数の分割数
        self.action_split_num = 10  # 行動数の分割数
        self.action_index = None
        self.state = (0, 0)
        self.state_old = (0, 0)
        self.reward = 0

        self.action_list = np.linspace(-1, 1, self.action_split_num)
        # Q値の初期化
        self.Q_value = {}

        for i in range(self.state_split_num):
            for j in range(self.state_split_num):
                self.Q_value[(i, j)] = np.zeros_like(self.action_list)

    def value_to_index(self, x, min_value, max_value):
        return int((x - min_value) / (max_value - min_value) * self.state_split_num)

    def choose_action(self):
        # ε-greedy法によって行動選択を行う
        if np.random.random() > self.epsiron:
            # 行動価値が最大の行動を選択

            self.action_index = np.argmax(self.Q_value[self.state])
        else:
            # ランダムな行動を選択
            self.action_index = np.random.choice(range(self.action_split_num))
        return self

    def set_motor(self):
        # ser.write(self.action_list[self.action_index])
        return self

    def update_state(self):
        self.state_old = self.state

        state = np.random.uniform(-0.5, 0.5, size=2)
        s1 = self.value_to_index(state[0], -1, 1)
        s2 = self.value_to_index(state[1], -1, 1)

        self.state = (s1, s2)
        return self

    def get_reward(self):
        self.reward = 10 - np.dot(self.state, self.state)
        return self

    def update_Qvalue(self):

        q_value = self.Q_value[self.state_old][self.action_index]
        q_value_max = np.max(self.Q_value[self.state])

        self.Q_value[self.state][self.action_index] = q_value + self.alpha * (
            self.reward + self.gamma * q_value_max - q_value)

        return self


if __name__ == "__main__":
    crawl = Crawl()

    while True:
        # 現在のQ値に従って行動選択
        u = crawl.choose_action()

        # クロールに行動指令
        crawl.set_motor()

        # クロールの状態を観測
        crawl.update_state()

        # 報酬を観測
        crawl.get_reward()

        # 行動前の状態における行動価値関数の更新
        crawl.update_Qvalue()
