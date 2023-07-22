import json5

import numpy as np
from physics import v_of_E_kin, E_kin_of_v


class ParticleType:
    def __init__(self, name, z, M):
        self.__name = name  # particle's name
        self.__z = z  # particle's charge
        self.__M = M  # MeV is a particle's mass

    @property
    def name(self): return self.__name

    @property
    def z(self): return self.__z

    @property
    def M(self): return self.__M


class ParticleTypes:
    __VALUES = {}

    @staticmethod
    def init():
        with open("particles.json5", "r") as f:
            particle_types = json5.load(f)

        for name, val in particle_types.items():
            ParticleTypes.__VALUES[name] = ParticleType(name, val["z"], val["M"])

    @staticmethod
    def by_name(name):
        return ParticleTypes.__VALUES[name]


class Particle:
    def __init__(self, particle_type: ParticleType):
        # Unwrapping the particle type
        self.name = particle_type.name
        self.z = particle_type.z
        self.M = particle_type.M
        self.mass = 1.7826627e-30 * particle_type.M  # kg is a particle's mass
        self.pos = np.array([0, 0, 0])
        self.vel = np.array([0, 0, 0])
        self.changed = False

    def set_pos(self, pos):
        self.pos = pos
        return self

    def set_vel(self, vel):
        self.vel = np.array(vel)
        return self

    def set_impulse(self, impulse):
        self.vel = impulse / self.mass
        return self

    def get_impulse(self):
        return self.vel * self.mass

    def set_kinetic_energy(self, E_kin, direction):
        self.vel = np.array(direction) * (v_of_E_kin(E_kin, self.mass) / np.linalg.norm(direction))
        return self

    def get_kinetic_energy(self):
        return E_kin_of_v(np.linalg.norm(self.vel), self.mass)

    def rotate_x(self, rot):
        self.vel[1] = self.vel[1] * np.cos(rot) - self.vel[2] * np.sin(rot)
        self.vel[2] = self.vel[1] * np.sin(rot) + self.vel[2] * np.cos(rot)
        return self

    def rotate_y(self, rot):
        self.vel[0] = self.vel[0] * np.cos(rot) + self.vel[2] * np.sin(rot)
        self.vel[2] = -self.vel[0] * np.sin(rot) + self.vel[2] * np.cos(rot)
        return self

    def rotate_z(self, rot):
        self.vel[0] = self.vel[0] * np.cos(rot) - self.vel[1] * np.sin(rot)
        self.vel[1] = self.vel[0] * np.sin(rot) + self.vel[1] * np.cos(rot)
        return self

    def rotate(self, x_rot, y_rot, z_rot):
        return self.rotate_x(x_rot).rotate_y(y_rot).rotate_z(z_rot)
