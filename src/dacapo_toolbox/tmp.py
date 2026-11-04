from funlib.geometry import Coordinate


def int_gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def gcd(a: Coordinate, b: Coordinate) -> Coordinate:
    return Coordinate(int_gcd(x, y) for x, y in zip(a, b, strict=False))
