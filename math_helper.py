import numpy as np
from numpy.random import RandomState


eps = 2.0 * np.finfo(float).eps  # machine epsilon


def lerp(a, b, f=0.5):
	return a + f * (b - a)


def gen_rand_direction(rand: RandomState = None):
	if rand is None:
		rand = RandomState()
	vec = rand.rand(3) - 0.5
	return vec / np.linalg.norm(vec)


# Runge-Kutta method
def rk4(max_time, delta_t, acceleration, t_0, p_0, v_0, event=None):
	if event is None:
		event = lambda t, dt, p, dp, v, dv: True

	# initial conditions
	t = t_0
	p = np.array(p_0)
	v = np.array(v_0)

	t_arr = [t]
	p_arr = [p]
	v_arr = [v]

	for _ in range(0, int(max_time // delta_t)):
		t += delta_t

		p1 = p
		v1 = v
		a1 = delta_t * acceleration(t, p1, v1)
		v1 = delta_t * v1

		p2 = p + (v1 * 0.5)
		v2 = v + (a1 * 0.5)
		a2 = delta_t * acceleration(t, p2, v2)
		v2 *= delta_t

		p3 = p + (v2 * 0.5)
		v3 = v + (a2 * 0.5)
		a3 = delta_t * acceleration(t, p3, v3)
		v3 *= delta_t

		p4 = p + v3
		v4 = v + a3
		a4 = delta_t * acceleration(t, p4, v4)
		v4 *= delta_t

		dv = a1 + 2.0 * (a2 + a3) + a4
		v = v + dv / 6.0

		dp = v1 + 2.0 * (v2 + v3) + v4
		p = p + dp / 6.0

		t_arr.append(t)
		p_arr.append(p)
		v_arr.append(v)

		if not event(t, delta_t, p, dp, v, dv):
			return t_arr, p_arr, v_arr

	return t_arr, p_arr, v_arr
