import numpy as np
from absorber import Absorbers, Absorber
from math_helper import lerp, gen_rand_direction
from particle import Particle, ParticleTypes
from physics import MeV2J, E_kin_of_v, compensate_impulse_3_2, v_of_E_kin, J2MeV


class NuclearReaction:
	@staticmethod
	def _none(total_impulse=0.0, rand: np.random.RandomState = None): return []

	@staticmethod
	def _proton_boron(total_impulse, rand: np.random.RandomState = None): 
		particle_1 = Particle(ParticleTypes.by_name("AlphaParticle"))
		particle_2 = Particle(ParticleTypes.by_name("AlphaParticle"))
		particle_3 = Particle(ParticleTypes.by_name("AlphaParticle"))
		impulse_1 = v_of_E_kin(MeV2J(0.9), particle_1.mass) * particle_1.mass * gen_rand_direction(rand)
		impulse_2, impulse_3 = compensate_impulse_3_2(total_impulse, impulse_1, v_of_E_kin(MeV2J(3.9), particle_2.mass) * particle_2.mass, rand)
		particle_1.set_impulse(impulse_1)
		particle_2.set_impulse(impulse_2)
		particle_3.set_impulse(impulse_3)
		return [particle_1, particle_2, particle_3]

	_DICT = {"Proton": {"Boron": _proton_boron}}

	def __init__(self, new_particles: [Particle], time, pos):
		self.new_particles = [particle.set_pos(pos) for particle in new_particles]
		self.time = time
		self.pos = pos
		self.impulse = sum([particle.get_impulse() for particle in new_particles])
		self.E_kin = sum([particle.get_kinetic_energy() for particle in new_particles])

	@classmethod
	def of(cls, particle: Particle, absorber: Absorber, time, pos, total_impulse, rand: np.random.RandomState = None):
		return NuclearReaction(NuclearReaction._from_dict(particle, absorber, total_impulse, rand), time, pos)

	@classmethod
	def none(cls, time, pos, total_impulse, rand: np.random.RandomState = None):
		return NuclearReaction(NuclearReaction._none(total_impulse, rand), time, pos)

	@staticmethod
	def exists(particle, absorber):
		return particle.name in NuclearReaction._DICT.keys() and absorber.name in NuclearReaction._DICT[particle.name].keys()

	@staticmethod
	def _from_dict(particle, absorber, total_impulse, rand: np.random.RandomState = None):
		if NuclearReaction.exists(particle, absorber):
			return NuclearReaction._DICT[particle.name][absorber.name](total_impulse, rand)
		return NuclearReaction._none(total_impulse, rand)

	@staticmethod
	def test(t_arr, p_arr, v_arr, absorber, particle, n_particles, cross_section_getter, rand: np.random.RandomState = None, probability_amplifier=1.0):
		if not NuclearReaction.exists(particle, absorber):
			return []
		assert len(t_arr) == len(p_arr) == len(v_arr)
		if rand is None:
			rand = np.random.RandomState()
		nuclear_reactions = []
		for (t_prev, t), (p_prev, p), (v_prev, v) in zip(zip(t_arr[0:], t_arr[1:]), zip(p_arr[0:], p_arr[1:]), zip(v_arr[0:], v_arr[1:])):
			probability = NuclearReaction.get_probability(absorber, particle, cross_section_getter, lerp(p_prev, p), lerp(v_prev, v), t - t_prev, probability_amplifier)
			rand_arr = rand.rand(n_particles)
			new_nuclear_reactions = [NuclearReaction.of(particle, absorber, lerp(t_prev, t), lerp(p_prev, p), particle.mass * lerp(v_prev, v), rand) for r in rand_arr if r <= probability]
			n_particles -= len(new_nuclear_reactions)
			nuclear_reactions += new_nuclear_reactions
		return nuclear_reactions

	@staticmethod
	def get_probability(absorber, particle, cross_section_getter, pos, vel, delta_t, amplifier=1.0):
		if not absorber.is_inside(pos):
			return 0.0
		vn = np.linalg.norm(vel)
		return amplifier * cross_section_getter(E_kin_of_v(vn, particle.mass)) * vn * absorber.atom_density * delta_t

	@staticmethod
	def rotate_randomly(particles: [Particle], rand: np.random.RandomState = None):
		if rand is None:
			rand = np.random.RandomState()
		rotations = (2 * rand.rand(3) - 1.0) * np.pi
		return [particle.rotate(rotations[0], rotations[1], rotations[2]) for particle in particles]

	def print_info(self):
		print("Nuclear reaction occurred")
		print(f"\tat Time: {self.time} s")
		print(f"\tat Position: {self.pos} [m]")
		print(f"\twith kinetic Energy: {J2MeV(self.E_kin * 1e3)} keV")
