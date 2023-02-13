import numpy as np
from numpy import testing as npt
from pytest import fixture

from yaw.core import coordinates


def test_position_sky2sphere():
    # test right ascension
    input = np.deg2rad([
        [-90., 0.],
        [  0., 0.],
        [ 90., 0.],
        [180., 0.],
        [270., 0.],
        [360., 0.],
        [450., 0.]])
    require = np.array([
        [ 0., -1.,  0.],
        [ 1.,  0.,  0.],
        [ 0.,  1.,  0.],
        [-1.,  0.,  0.],
        [ 0., -1.,  0.],
        [ 1.,  0.,  0.],
        [ 0.,  1.,  0.]])
    result = coordinates.position_sky2sphere(input)
    npt.assert_allclose(require, result, atol=1e-15)
    # test declination
    input = np.deg2rad([
        [-90.,  90.],
        [  0.,  90.],
        [ 90.,  90.],
        [-90., -90.],
        [  0., -90.],
        [ 90., -90.]])
    require = np.array([
        [0., 0.,  1.],
        [0., 0.,  1.],
        [0., 0.,  1.],
        [0., 0., -1.],
        [0., 0., -1.],
        [0., 0., -1.]])
    result = coordinates.position_sky2sphere(input)
    npt.assert_allclose(require, result, atol=1e-15)


def test_position_sphere2sky():
    # test right ascension
    input = np.array([
        [ 1.,  0.,  0.],
        [ 0.,  1.,  0.],
        [-1.,  0.,  0.],
        [ 0., -1.,  0.]])
    require = np.deg2rad([
        [  0., 0.],
        [ 90., 0.],
        [180., 0.],
        [270., 0.]])
    result = coordinates.position_sphere2sky(input)
    npt.assert_allclose(require, result, atol=1e-15)
    # test declination
    input = np.array([
        [0., 0.,  1.],
        [0., 0., -1.]])
    require = np.deg2rad([
        [0.,  90.],
        [0., -90.]])
    result = coordinates.position_sphere2sky(input)
    npt.assert_allclose(require, result, atol=1e-15)


@fixture
def dist_sky():
    return np.pi * np.array([0.0, 0.5, 1.0])


@fixture
def dist_sphere():
    return np.array([0.0, np.sqrt(2.0), 2.0])


def test_distance_sky2sphere(dist_sky, dist_sphere):
    npt.assert_allclose(coordinates.distance_sky2sphere(dist_sky), dist_sphere)



def test_distance_sphere2sky(dist_sky, dist_sphere):
    npt.assert_allclose(coordinates.distance_sphere2sky(dist_sphere), dist_sky)
