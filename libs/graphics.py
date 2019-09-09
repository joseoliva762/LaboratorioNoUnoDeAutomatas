import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from pylab import *
from sklearn.model_selection import train_test_split as tts

class Graphics():
    def __init__(self):
        pass

    def show_img(self):
        plt.show()

    def graphics(self, labelx, labely, obo=True):
        for dims in labelx:
            if plt.scatter(labelx[dims], labely):
                if obo == True:
                    self.show_img()
                print('[OK]    Grafics from {}.'.format(dims))
            else:
                print('[FAIL]  Grafics from {}.'.format(dims))
        if obo == False:
            self.show_img()

    def graphics_with_plot(self, ep, train_cost):
        plt.plot(ep, train_cost)
        self.show_img()
