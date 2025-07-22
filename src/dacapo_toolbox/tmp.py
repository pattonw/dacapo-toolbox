from funlib.geometry import Coordinate
import numpy as np
from fractions import Fraction
from math import lcm


def int_gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def gcd(a: Coordinate, b: Coordinate) -> Coordinate:
    return Coordinate([int_gcd(x, y) for x, y in zip(a, b)])


def int_scale(coords: list[Coordinate]) -> list[Coordinate]:
    # Get per-dimension scale factors
    X = np.array(coords)
    scales = []
    for dim in range(X.shape[1]):
        denominators = []
        for x in X[:, dim]:
            frac = Fraction(x).limit_denominator(100)  # adjust max denominator as needed
            denominators.append(frac.denominator)
        scales.append(lcm(*denominators))

    scale_vector = np.array(scales)
    return Coordinate(scale_vector)