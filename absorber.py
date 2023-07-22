# from scipy.integrate import odeint
import json5

import numpy as np
from numpy import log, pi
from math_helper import rk4
from particle import Particle
from physics import c, beta, m_e, eps_0, q_e, N_A, MeV2J


class Absorber:
	particle = None
	__C1 = None
	__C2 = None
	__num_points = 10000
	__flight_time = 0.0
	delta_t = 1e-8

	# Using of construcor outside the Absorbers class is usafe!
	def __init__(self, name, Z, A, I, density, is_inside=lambda p: True):
		self.name = name
		self.Z = Z
		self.A = A
		self.I = I
		self.density = density  # kg m^-3 is a density of material
		self.atom_mass = A / (N_A * 1e3)
		self.atom_density = density / self.atom_mass
		self.electron_density = Z * self.atom_density
		self.is_inside = is_inside

	def set_particle(self, particle):
		self.particle = particle
		self.__update_constants()

	def __update_constants(self):
		self.__C1 = 4 * pi * self.electron_density * self.particle.z ** 2 * q_e ** 4 / (m_e * (4 * pi * eps_0 * c) ** 2)
		self.__C2 = 2 * m_e * c ** 2 / self.I

	# Energy loss in J m^-1
	def dE_over_dx(self, v):
		if v < 0.001 * c:
			return self.dE_over_dx_low(v)
		sqr_beta = beta(v) ** 2
		return -self.__C1 / sqr_beta * (log(self.__C2 * sqr_beta) - log(1 - sqr_beta) - sqr_beta)

	def dE_over_dx_low(self, v):
		return -self.__C1 * (c / v) ** 2 * (log(self.__C2 / (c * v) ** 2))

	def dE_over_dt(self, v):
		return self.dE_over_dx(v) * v

	def acceleration(self, p, v):
		if not self.is_inside(p):
			return np.zeros(3)
		vn = np.linalg.norm(v)
		if vn < 0.001 * c:
			return self.dE_over_dx_low(vn) / self.particle.mass
		return v * ((1.0 - beta(v) ** 2) ** 1.5 / (self.particle.mass * vn) * self.dE_over_dx(vn))

	# def E_loss_vs_x(self, x_arr, particle: Particle):
	#     E_kin_0 = particle.get_kinetic_energy()
	#     self.set_particle(particle)
	#     return np.array(odeint(lambda E_kin, x: self.dE_over_dx(v_of_E_kin(E_kin, self.particle.mass)), E_kin_0, x_arr, rtol=1e-5)).flatten()
	# 
	# def E_loss_vs_t(self, t_arr, particle: Particle):
	#     E_kin_0 = particle.get_kinetic_energy()
	#     self.set_particle(particle)
	#     return np.array(odeint(lambda E_kin, t: self.dE_over_dt(v_of_E_kin(E_kin, self.particle.mass)), E_kin_0, t_arr, rtol=1e-5)).flatten()

	def run(self, t_0, max_time, delta_t, particle: Particle, event=None):
		self.set_particle(particle)
		if event is None:
			event = lambda t, dt, p, dp, v, dv: not np.allclose(dv, [0.0, 0.0, 0.0])
		return rk4(max_time, delta_t, lambda t, p, v: self.acceleration(p, v), t_0, particle.pos, particle.vel, event)


class Absorbers:
	__VALUES = {}

	@staticmethod
	def init():
		with open("absorbers.json5", "r") as f:
			particle_types = json5.load(f)

		for name, val in particle_types.items():
			Absorbers.__VALUES[name] = Absorber(name, val["Z"], val["A"], MeV2J(val["I"]), val["density"])

	@staticmethod
	def by_name(name):
		return Absorbers.__VALUES[name]
