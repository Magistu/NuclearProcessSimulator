import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import linregress

student_coefficient = 2.1447866879169273


class ExperimentalData:
    def __init__(self, name, x_name, y_name, x_var, y_var, x_data, y_data):
        self.name = name
        self.x_name = x_name
        self.y_name = y_name
        self.x_var = x_var
        self.y_var = y_var
        self.x_data = x_data
        self.y_data = y_data
        self.x_linspace = np.linspace(min(x_data), max(x_data), num=1 * len(x_data))
        self.linreg = None
        self.y_linreg = None
        self.linreg_expression = None
        self.interpol = None
        self.y_interpol = None

    @classmethod
    def from_file(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            plot_name = Path(f.name).stem
            x_data = []
            y_data = []
            for i, line in enumerate(f):
                if i == 0:
                    x_name = line.replace('\n', '')
                    continue
                if i == 1:
                    y_name = line.replace('\n', '')
                    continue
                if i == 2:
                    x_var = line.replace('\n', '')
                    continue
                if i == 3:
                    y_var = line.replace('\n', '')
                    continue
                line = re.sub(r';|\t', ' ', line)
                line = line.replace(',', '.')
                x_y = list(map(float, re.split(r' +', line)))
                x_data.append(x_y[0])
                y_data.append(x_y[1])
            return ExperimentalData(plot_name, x_name, y_name, x_var, y_var, x_data, y_data)

    def linregress(self):
        self.linreg = linregress(self.x_data, self.y_data)
        self.linreg_expression = ExperimentalData.to_expression(self.x_var, self.y_var, self.linreg)
        linreg_func = ExperimentalData.to_func(self.linreg)
        self.y_linreg = np.array([linreg_func(x) for x in self.x_linspace])

    def interpolate(self):
        self.interpol = interp1d(self.x_data, self.y_data, kind='cubic')
        self.y_interpol = np.array([self.interpol(x) for x in self.x_linspace])

    def apply_func_on_y(self, func, y_name, y_var):
        self.y_name = y_name
        self.y_var = y_var
        self.y_data = np.array([func(y) for y in self.y_data])

    def plt(self, linreg=True, interpol=True, color=None, xlim=None, ylim=None):
        if linreg and self.linreg is not None:
            plt.plot(self.x_linspace, self.y_linreg, color=color)
        if interpol and self.interpol is not None:
            plt.plot(self.x_linspace, self.y_interpol, color=color)
        plt.plot(self.x_data, self.y_data, 'o', color=color, label=self.name)
        plt.title(self.name)
        plt.xlabel(self.x_name)
        plt.ylabel(self.y_name)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

    def save(self, save_dir='', build_plot=True):
        table = ExperimentalData.to_table(self.x_data, self.linreg, decimal_comma=True)
        with open(save_dir + self.name + '.txt', 'w', encoding='utf-8') as f:
            f.write(table)
        if build_plot:
            self.plt(linreg=False)
            plt.savefig(save_dir + self.name + '.png')
            plt.close()

    def save_linreg(self, save_dir=''):
        if save_dir != '':
            save_dir += '/'
        self.linregress()
        table = ExperimentalData.to_table(self.x_linspace, self.y_linreg, decimal_comma=True)
        results_data = self.linreg_expression + '\n\n' + table
        sys.stdout.buffer.write(results_data.encode('utf-8'))
        with open(save_dir + self.name + '.txt', 'w', encoding='utf-8') as f:
            f.write(results_data)

    def save_interpol(self, save_dir=''):
        if save_dir != '':
            save_dir += '/'
        self.interpolate()
        table = ExperimentalData.to_table(self.x_linspace, self.y_interpol, decimal_comma=True)
        sys.stdout.buffer.write(table.encode('utf-8'))
        with open(save_dir + self.name + '.txt', 'w', encoding='utf-8') as f:
            f.write(table)

    def get_y(self, x_0):
        return np.interp(x_0, self.x_data, self.y_data)

    @staticmethod
    def to_table(x_data, y_data, decimal_comma=False):
        table = ''
        for x, y in zip(x_data, y_data):
            table += '%f\t%f\n' % (x, y)
        if decimal_comma:
            table = table.replace('.', ',')
        return table

    @staticmethod
    def to_func(result):
        def func(x):
            return result.slope * x + result.intercept
        return func

    @staticmethod
    def to_expression(x_name, y_name, result):
        return '%s = (%f ± %f) * %s %s (%f ± %f)' % (
            y_name,
            result.slope,
            student_coefficient * result.stderr,
            x_name,
            '+' if result.intercept > 0 else '-' if result.intercept < 0 else '',
            math.fabs(result.intercept),
            student_coefficient * result.intercept_stderr)
