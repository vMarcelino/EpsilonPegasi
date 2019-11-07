import numpy as np
import random
import math
from dataclasses import dataclass
from epsilon_pegasi.helpers import enforced_dataclass
from epsilon_pegasi.camera import Camera
from epsilon_pegasi.shapes import Scene
from epsilon_pegasi.base_classes import Ray

typing_replaces = {np.ndarray: lambda x: np.array(x, dtype=float)}


class Color3(np.ndarray):
    pass


@enforced_dataclass(replaces=typing_replaces)
class RenderOptions:
    width: int
    height: int
    maximumDepth: int
    cameraSamples: int
    lightSamples: int
    diffuseSamples: int
    filterWidth: float
    gamma: float
    exposure: float


@dataclass
class Renderer:
    options: RenderOptions
    camera: Camera
    scene: Scene

    def stratifiedSample(self, samples: int) -> np.ndarray:
        size = int(np.sqrt(samples))
        points = [[0, 0] for i in range(samples)]

        for i in range(size):
            for j in range(size):
                offset = np.array([i, j])
                points[i * size + j] = (offset + [random.uniform(0, 1), random.uniform(0, 1)]) / size

        return points

    def gamma(self, color: Color3, value: float) -> Color3:
        return color / value

    def exposure(self, color: Color3, value: float) -> Color3:
        power = 2**value

        return color * power

    def saturate(self, x: Color3) -> Color3:
        return np.clip(x, 0, 1)

    def trace(self, ray: Ray, depth: int) -> Color3:
        intersection = self.scene.intersects(ray)

        if intersection.hit:
            return np.array([1.0, 1.0, 1.0])

        return np.array([0.0, 0.0, 0.0])

    def doge_render(self) -> np.ndarray:
        result = np.zeros((int(self.camera.width), int(self.camera.height), 3))
        for i in range(int(self.camera.width)):
            for j in range(int(self.camera.height)):
                ray = self.camera.generateRay(i, j)
                intersection = self.scene.intersects(ray)
                result[i][j] = np.where(intersection.hit, (1, 1, 1), (0, 0, 0))

        return result

    def willerBrener_render(self) -> np.ndarray:
        image = np.zeros((int(self.options.width), int(self.options.height), 3))
        for i in range(self.options.width):
            for j in range(self.options.height):
                samples: np.ndarray = self.stratifiedSample(self.options.cameraSamples)

                color = np.array([0, 0, 0], dtype=float)
                totalWeight = 0

                for k in range(self.options.cameraSamples):
                    sample = (samples[k] - [0.5, 0.5]) * self.options.filterWidth
                    ray = self.camera.generateRay(i, j, sample)
                    weight = gaussian2D(sample, self.options.filterWidth)

                    color += self.trace(ray, 0) * weight
                    totalWeight += weight

                color /= totalWeight
                image[i][j] = self.saturate(self.gamma(self.exposure(color, self.options.exposure), self.options.gamma))
        return image


def gaussian2D(X, w):
    r = w / 2
    k = 1
    for K in [max(math.exp(-(x**2)) - math.exp(-(r**2)), 0) for x in X]:
        k *= K
    return k
