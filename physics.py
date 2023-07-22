from math import acos
from numpy import sqrt, ndarray, sin, cross
import numpy as np
from numpy.random import RandomState

from math_helper import eps, gen_rand_direction
import matplotlib.pyplot as plt


m_e = 9.109e-31  # kg is the electron mass
q_e = 1.602e-19  # C is the electron charge
N_A = 6.022e23  # is Avogadro constant
eps_0 = 8.854e-12  # F/m is the permittivity of vacuum
c = 3e8  # m/s is the speed of light
h = 6.62607015e-34  # Planck's constant


def beta(v): return v / c


def MeV2J(E): return 1e6 * q_e * E


def J2MeV(E): return E / (1e6 * q_e)


def v_of_E_kin(E_kin, mass):
	if E_kin < eps:
		return eps
	sqr_v = (2 * E_kin) / mass
	if sqr_v < (0.001 * c) ** 2:
		return sqrt(sqr_v)
	return sqrt(max(eps, 1.0 - 1.0 / (1.0 + E_kin / (mass * c ** 2)) ** 2)) * c


def E_kin_of_v(v, mass):
	if v < 0.001 * c:
		return 0.5 * mass * v ** 2
	return mass * c ** 2 * (1.0 / sqrt(1.0 - beta(v) ** 2) - 1.0)


def abs_inelastic_collision(m_1, v_1, m_2, v_2):
	m = m_1 + m_2
	return m, (m_1 * v_1 + m_2 * v_2) / m


# find impulse_2_vec and impulse_3_vec to compensate total_impulse_vec - impulse_1_vec considering that impulse_2 = impulse_3 = impulse_2_3
def compensate_impulse_3_2(total_impulse_vec: ndarray, impulse_1_vec: ndarray, impulse_2_3: float, rand: RandomState = None):
	if rand is None:
		rand = RandomState()
	impulse_2_3_sum_vec = total_impulse_vec - impulse_1_vec
	impulse_2_3_sum = np.linalg.norm(impulse_2_3_sum_vec)
	impulse_2_3_sum_dir = impulse_2_3_sum_vec / impulse_2_3_sum
	impulse_2_3_sum_left_vec = cross(impulse_2_3_sum_dir, gen_rand_direction(rand))
	impulse_2_3_sum_left_dir = impulse_2_3_sum_left_vec / np.linalg.norm(impulse_2_3_sum_left_vec)
	half_angle_2_3 = acos(impulse_2_3_sum / (2.0 * impulse_2_3))
	impulse_2_vec = impulse_2_3_sum_vec / 2.0 + sin(half_angle_2_3) * impulse_2_3_sum * impulse_2_3_sum_left_dir
	impulse_3_vec = impulse_2_3_sum_vec / 2.0 - sin(half_angle_2_3) * impulse_2_3_sum * impulse_2_3_sum_left_dir
	return impulse_2_vec, impulse_3_vec


def test():
	total_impulse_vec = 0.6 * np.array([1.0, 0.0, 0.0])
	impulse_1_vec = 0.9 * np.array([1.0 / sqrt(2), 1.0 / sqrt(2), 0.0])
	impulse_2_3 = 3.9
	impulse_2_vec, impulse_3_vec = compensate_impulse_3_2(total_impulse_vec, impulse_1_vec, impulse_2_3)
	plt.plot([0.0, total_impulse_vec[0]], [0.0, total_impulse_vec[1]], color='r')
	plt.plot([0.0, impulse_1_vec[0]], [0.0, impulse_1_vec[1]], color='g')
	plt.plot([0.0, impulse_2_vec[0]], [0.0, impulse_2_vec[1]], color='b')
	plt.plot([0.0, impulse_3_vec[0]], [0.0, impulse_3_vec[1]], color='b')
	plt.show()
	
	plt.plot([0.0, total_impulse_vec[2]], [0.0, total_impulse_vec[1]], color='r')
	plt.plot([0.0, impulse_1_vec[2]], [0.0, impulse_1_vec[1]], color='g')
	plt.plot([0.0, impulse_2_vec[2]], [0.0, impulse_2_vec[1]], color='b')
	plt.plot([0.0, impulse_3_vec[2]], [0.0, impulse_3_vec[1]], color='b')
	plt.show()
	
	
if __name__ == "__main__":
	test()
