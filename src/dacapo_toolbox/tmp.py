from funlib.geometry import FloatCoordinate


def int_gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def gcd(a: FloatCoordinate[int], b: FloatCoordinate[int]) -> FloatCoordinate[int]:
    return FloatCoordinate(int_gcd(x, y) for x, y in zip(a, b))
