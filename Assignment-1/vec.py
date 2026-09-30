import math
import random
from typing import Self

"""
A custom vector class implementation for educational purposes.
"""


class Vec:
    def __init__(self, src=None) -> Self:
        if src is None:
            self.elements = ()
        else:
            elements = tuple(src)

            for x in elements:
                if not isinstance(x, (int, float)):
                    raise TypeError(f"Scalar must be a number: {type(x)}")

            self.elements = elements

    def __add__(self, t: Self) -> Self:
        if not isinstance(t, Vec):
            raise TypeError(f"Expected Vec: {type(t)}")

        if len(self.elements) != len(t):
            raise TypeError("Type error - vectors must be of same dimensions")

        result = []

        for x, y in zip(self.elements, t.elements):
            result.append(round(x + y, 5))

        return Vec(tuple(result))

    def __rmul__(self, scalar: int | float) -> Self:
        if not isinstance(scalar, (int, float)):
            raise TypeError(
                f"Vector multiplication with invalid type: {type(scalar)}"
            )

        result = []

        for value in self.elements:
            result.append(round(value * scalar, 5))

        return Vec(tuple(result))

    def __imul__(self, scalar: int | float) -> Self:
        if not isinstance(scalar, (int, float)):
            raise TypeError(
                f"Vector multiplication with invalid type: {type(scalar)}"
            )

        result = []

        for value in self.elements:
            result.append(round(value * scalar, 5))

        self.elements = tuple(result)

        return self

    def __repr__(self) -> str:
        return repr(self.elements)

    def __len__(self) -> int:
        return len(self.elements)

    def __sub__(self, t: Self) -> Self:
        if not isinstance(t, Vec):
            raise TypeError(f"Expected Vec: {type(t)}")

        if len(self.elements) != len(t):
            raise TypeError("Type error - vectors must be of same dimensions")

        result = []

        for x, y in zip(self.elements, t.elements):
            result.append(round(x - y, 5))

        return Vec(tuple(result))

    def __neg__(self) -> Self:
        result = []

        for value in self.elements:
            result.append(-value)

        return Vec(tuple(result))

    def __radd__(self, other):
        if not isinstance(other, Vec):
            raise TypeError(f"Expected Vec: {type(other)}")

        return other + self

    def __iadd__(self, other):
        if not isinstance(other, Vec):
            raise TypeError(f"Expected Vec: {type(other)}")

        if len(self.elements) != len(other.elements):
            raise TypeError("Type error - vectors must be of same dimensions")

        result = []

        for x, y in zip(self.elements, other.elements):
            result.append(round(x + y, 5))

        self.elements = tuple(result)

        return self

    @staticmethod
    def zeros(n: int) -> Self:
        if n <= 0:
            raise ValueError("Vector dimension must be greater than zero")

        result = []

        for _ in range(n):
            result.append(0)

        return Vec(tuple(result))

    @staticmethod
    def ones(n: int) -> Self:
        if n <= 0:
            raise ValueError("Vector dimension must be greater than zero")

        result = []

        for _ in range(n):
            result.append(1)

        return Vec(tuple(result))

    @staticmethod
    def uniform(n: int) -> Self:
        if n <= 0:
            raise ValueError("Vector dimension must be greater than zero")

        result = []

        for _ in range(n):
            result.append(random.random())

        return Vec(tuple(result))

    def norm(self) -> float:
        total = 0

        for value in self.elements:
            total += value * value

        return math.sqrt(total)

    # Assignment1

    def mean(self) -> float:
        return sum(self.elements) / len(self.elements)

    def demean(self) -> Self:
        average = self.mean()
        result = []

        for value in self.elements:
            result.append(value - average)

        return Vec(tuple(result))

    def std(self) -> float:
        demeaned = self.demean()
        total = 0

        for value in demeaned.elements:
            total += value * value

        return math.sqrt(total / len(demeaned))


if __name__ == "__main__":
    v1 = Vec((0, 1, 1.03))

    print("Vector:", v1)
    print("Norm:", v1.norm())