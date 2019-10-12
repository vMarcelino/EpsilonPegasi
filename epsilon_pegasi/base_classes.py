from dataclasses import dataclass
from epsilon_pegasi.helpers import enforced_dataclass
from typing import List
from enum import Enum
import numpy as np


@enforced_dataclass
class Ray:
    origin: np.array
    direction: np.array

    def point(self, distance: float) -> np.ndarray:
        pass


@dataclass
class Color3:
    r: int
    g: int
    b: int


class BSDFType(Enum):
    light = 0
    diffuse = 1
    specular = 2
    none = 3


@dataclass
class BSDF:
    type: BSDFType
    color: Color3


@enforced_dataclass
class ShaderGlobals:
    point: np.array
    normal: np.array
    uv: np.ndarray
    tangentU: np.array
    tangentV: np.array
    viewDirection: np.array
    lighDirection: np.array
    lightPoint: np.array
    lighNormal: np.array


@enforced_dataclass
class Vertex:
    position: np.array
    normal: np.array = np.array([0, 0])
    uv: np.array = np.array([0, 0])
