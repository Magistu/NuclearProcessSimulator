import json5

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState

from data_cls import ExperimentalData
from absorber import Absorbers
from nuclear_reaction import NuclearReaction
from particle import Particle, ParticleTypes
from physics import J2MeV, MeV2J, v_of_E_kin

MENU_TEXT = """enter:
0 to run with parameters specified in parameters.json5
exit to exit
"""


def run():
    with open("parameters.json5", "r") as f:
        parameters = json5.load(f)

    max_time = parameters["maxTime"]
    absorber = Absorbers.by_name(parameters["absorber"])
    boundary_x = parameters["boundaryX"]
    particle_type = ParticleTypes.by_name(parameters["particleType"])
    E_kin_0 = MeV2J(parameters["EKin0"])
    n_particles = parameters["nParticles"]
    probability_amplifier = parameters["probabilityAmplifier"]
    seed = parameters["seed"]
    if seed is None:
        seed = np.random.randint(0, 1000000)
    n_points = parameters["nPoints"]
    sgm_file = parameters["sgmFile"]
    nuclear_reaction_print = parameters["nuclearReactionPrint"]

    particle = Particle(particle_type).set_kinetic_energy(E_kin_0, [1.0, 0.0, 0.0])
    sgm_data = ExperimentalData.from_file(sgm_file)
    absorber.is_inside = lambda p: p[0] > boundary_x
    rand = RandomState(seed)

    print(f"Absorber: {absorber.name}")
    print(f"Number of {particle.name}s: {n_particles}")
    print(f"Initial Energy: {J2MeV(particle.get_kinetic_energy() * 1e3)} keV")
    print(f"Probability amplifier: {probability_amplifier}")
    print(f"Boundary of absorber is at x = {boundary_x} m coordinate")
    print(f"Seed is {seed}")

    t_arr, p_arr, v_arr = absorber.run(0.0, max_time, max_time / n_points, particle,
                                       event=lambda t, dt, p, dp, v, dv: not absorber.is_inside(p) or not np.allclose(
                                           dv, [0.0, 0.0, 0.0]))

    nuclear_reactions = NuclearReaction.test(t_arr, p_arr, v_arr, absorber, particle, n_particles,
                                             lambda E_kin: 1e-28 * sgm_data.get_y(1e3 * J2MeV(E_kin)), rand,
                                             probability_amplifier)

    vn_arr = [np.linalg.norm(v) for v in v_arr]
    plt.plot(t_arr, vn_arr)
    prod_p_arr_arr = []

    n_out_of_absorber = 0
    for nuclear_reaction in nuclear_reactions:
        for prod_particle in nuclear_reaction.new_particles:
            prod_t_arr, prod_p_arr, prod_v_arr = absorber.run(nuclear_reaction.time, max_time, max_time / n_points,
                                                              prod_particle)
            prod_p_arr_arr.append(prod_p_arr)
            if abs(prod_p_arr[-1][0] - boundary_x) < 1e-10 or not absorber.is_inside(prod_p_arr[-1]):
                n_out_of_absorber += 1

        # # Check impulse conservation
        # print("Proton's impulse:", np.interp(nuclear_reaction.time, t_arr, vn_arr) * particle.mass)
        # print("New particles' impulse:", nuclear_reaction.impulse)

        if nuclear_reaction_print:
            nuclear_reaction.print_info()
        plt.axvline(x=nuclear_reaction.time, color="r", label="reaction case")

    plt.title(f"Velocity of {particle.name}s passing through {absorber.name}")
    plt.xlabel("Time [sec]")
    plt.ylabel("Velocity [m/s]")
    plt.savefig("velocity_vs_time.png")
    plt.close()

    if len(nuclear_reactions) > 0:
        for prod_p_arr in prod_p_arr_arr:
            prod_p_arr = np.array(prod_p_arr)
            plt.plot(prod_p_arr[:, 0], prod_p_arr[:, 1], color="r")
            plt.axis("equal")
            plt.title(f"Tracking of new particles in XY space")
            plt.xlabel("X-Dimension [m]")
            plt.ylabel("Y-Dimension [m]")
            plt.axvline(x=boundary_x, color="m", label=f"boundary of {absorber.name} absorber")
        plt.savefig("xy_tracking.png")
        plt.close()

        for prod_p_arr in prod_p_arr_arr:
            prod_p_arr = np.array(prod_p_arr)
            plt.plot(prod_p_arr[:, 2], prod_p_arr[:, 1], color="r")
            plt.axis("equal")
            plt.title(f"Tracking of new particles in ZY space")
            plt.xlabel("Z-Dimension [m]")
            plt.ylabel("Y-Dimension [m]")
        plt.savefig("zy_tracking.png")
        plt.close()

    print(f"Number of reactions that occurred is {len(nuclear_reactions)}")
    print(f"Number of particles that came out of absorber is {n_out_of_absorber}")


def test():
    absorber = Absorbers.by_name("Hydrogen")
    particle = Particle(ParticleTypes.by_name("AlphaParticle"))
    absorber.set_particle(particle)
    E_kin_0_arr = np.logspace(0, 3, 40)
    stopping_power_arr = [
        -J2MeV(absorber.dE_over_dx(v_of_E_kin(MeV2J(E_kin_0), particle.mass))) / absorber.density * 1e1 for E_kin_0 in
        E_kin_0_arr]
    print([[E_kin_0, stopping_power] for E_kin_0, stopping_power in zip(E_kin_0_arr, stopping_power_arr)])
    plt.loglog(E_kin_0_arr, stopping_power_arr)
    plt.title(f"{absorber.name}")
    plt.xlabel(f"Energy (MeV)")
    plt.ylabel(f"Stopping Power (MeV cm^2/g)")
    plt.savefig("stopping_power.png")
    plt.close()


def main():
    while True:
        state = input(MENU_TEXT)
        if state == "0":
            try:
                run()
                print("DONE")
            except Exception as e:
                print(e)
        elif state == "exit":
            break


if __name__ == "__main__":
    Absorbers.init()
    ParticleTypes.init()
    main()
