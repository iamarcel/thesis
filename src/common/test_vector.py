from nose.tools import ok_
import numpy as np

from . import vector


def test_shortest_angle_between():
  ok_(
      vector.shortest_angle_between(
          np.array([0., 1., 0.]), np.array([1., 0., 0.])) - np.pi / 2 < 1.0e-5)

  ok_(
      vector.shortest_angle_between(
          np.array([0., 1., 1.]), np.array([1., 0., 0.])) - np.pi / 2 < 1.0e-5)

  ok_(
      vector.shortest_angle_between(
          np.array([0., 1., 1.]), np.array([0., 1., 0.])) - np.pi / 4 < 1.0e-5)
