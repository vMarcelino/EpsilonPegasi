from dataclasses import dataclass
from epsilon_pegasi.helpers import enforced_dataclass
from typing import List
from enum import Enum
import numpy as np
typing_replaces = {np.ndarray: lambda x: np.array(x, dtype=float)}


@enforced_dataclass(replaces=typing_replaces)
class Ray:
    origin: np.ndarray
    direction: np.ndarray

    def point(self, distance: float) -> np.ndarray:
        pass


class BSDFType(Enum):
    light = 0
    diffuse = 1
    specular = 2
    none = 3


@dataclass
class BSDF:
    type: BSDFType
    color: np.ndarray


@enforced_dataclass(replaces=typing_replaces)
class ShaderGlobals:
    point: np.ndarray
    normal: np.ndarray
    uv: np.ndarray
    tangentU: np.ndarray
    tangentV: np.ndarray
    viewDirection: np.ndarray
    lighDirection: np.ndarray
    lightPoint: np.ndarray
    lighNormal: np.ndarray


@enforced_dataclass(replaces=typing_replaces)
class Vertex:
    position: np.ndarray
    normal: np.ndarray = np.array([0, 0])
    uv: np.ndarray = np.array([0, 0])
