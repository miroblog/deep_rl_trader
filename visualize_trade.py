import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def show_mdd(xs): # xs is cumulative return / portfolio , if reward u should
    # xs = df['reward'].cumsum() / if reward
    i = np.argmax(np.maximum.accumulate(xs) - xs) # end of the period
    j = np.argmax(xs[:i]) # start of period
    plt.plot(xs)
    plt.plot([i, j], [xs[i], xs[j]], 'o', color='Red', markersize=10)
    plt.show()

def visualize(info):
    closes = [data[2] for data in info['history']]
    closes_index = [data[1] for data in info['history']]
    # buy tick
    buy_tick = np.array([data[1] for data in info['history'] if data[0] == 0])
    buy_price = np.array([data[2] for data in info['history'] if data[0] == 0])
    sell_tick = np.array([data[1] for data in info['history'] if data[0] == 1])
    sell_price = np.array([data[2] for data in info['history'] if data[0] == 1])

    plt.plot(closes_index, closes)
    plt.scatter(buy_tick, buy_price - 2, c='g', marker="^", s=20)
    plt.scatter(sell_tick, sell_price + 2, c='r', marker="v", s=20)
    plt.show(block=True)
    time.sleep(3)

FILENAME = "duel_dqn_OHLCV-v0_weights_1688081LS_794_820_0.6099948635190375.info"
info = np.load(FILENAME).all()
visualize(info)