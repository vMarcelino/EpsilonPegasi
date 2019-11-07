import numpy as np
import random
import math
import numba
import sys
from dataclasses import dataclass
from epsilon_pegasi.helpers import enforced_dataclass
from epsilon_pegasi.camera import Camera, OptimizedCamera
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

    def trace(self, ray: Ray, depth: int) -> Color3:
        intersection = self.scene.intersects(ray)

        if intersection.hit:
            return np.array([1.0, 1.0, 1.0])

        return np.array([0.0, 0.0, 0.0])

    def willerBrener_render(self) -> np.ndarray:
        w, h = self.options.width, self.options.height
        image = np.zeros((h, w, 3))
        for i in range(h):
            for j in range(w):
                samples: np.ndarray = stratifiedSample(self.options.cameraSamples)

                color = np.array([0, 0, 0], dtype=float)
                totalWeight = 0

                for k in range(self.options.cameraSamples):
                    sample = (samples[k] - [0.5, 0.5]) * self.options.filterWidth
                    ray = self.camera.generateRay(j, i, sample)
                    weight = gaussian2D(sample, self.options.filterWidth)

                    color += self.trace(ray, 0) * weight
                    totalWeight += weight

                color /= totalWeight
                image[i][j] = saturate(gamma(exposure(color, self.options.exposure), self.options.gamma))
        return image


def gaussian2D(X, w):
    r = w / 2
    k = 1
    for K in [gaussian_1D(x, r) for x in X]:
        k *= K
    return k


@numba.njit
def gaussian_1D(x, r):
    return max(math.exp(-(x**2)) - math.exp(-(r**2)), 0)


@numba.njit
def gaussian_2D(x, y, w):
    r = w / 2
    return gaussian_1D(x, r) * gaussian_1D(y, r)


@numba.njit
def stratifiedSample(samples: int) -> np.ndarray:
    size = int(np.sqrt(samples))
    points = np.zeros((samples, 2))

    for i in range(size):
        for j in range(size):
            offset = np.array([i, j])
            rand = np.array((random.uniform(0, 1), random.uniform(0, 1)), numba.float32)
            points[i * size + j] = (offset + rand) / size

    return points


@numba.njit
def gamma(color: Color3, value: float) -> Color3:
    return color / value


@numba.njit
def exposure(color: Color3, value: float) -> Color3:
    power = 2**value

    return color * power


@numba.njit
def saturate(x: Color3) -> Color3:
    return np.clip(x, 0, 1)


def doge_render(options: RenderOptions, scene: Scene, camera: OptimizedCamera) -> np.ndarray:
    w = options.width
    h = options.height

    tris = [[vert.position for vert in tri.vertices] for tri in scene.shapes]

    return optimized_renderer(int(w), int(h), int(options.cameraSamples), options.filterWidth, options.exposure, options.gamma, camera,
                              tris)
    # image = np.zeros((h, w, 3))
    # for i in range(h):
    #     for j in range(w):
    #         samples: np.ndarray = self.stratifiedSample(self.options.cameraSamples)

    #         color = np.array([0, 0, 0], dtype=float)
    #         totalWeight = 0

    #         for k in range(self.options.cameraSamples):
    #             sample = (samples[k] - [0.5, 0.5]) * self.options.filterWidth
    #             ray = self.camera.generateRay(j, i, sample)
    #             weight = gaussian_2D(sample[0], sample[1], self.options.filterWidth)

    #             color += self.trace(ray, 0) * weight
    #             totalWeight += weight

    #         color /= totalWeight
    #         image[i][j] = self.saturate(self.gamma(self.exposure(color, self.options.exposure), self.options.gamma))
    # return image


@numba.njit
def optimized_renderer(w, h, sample_count, filterWidth, exposure, gamma, camera: OptimizedCamera, scene_triangles):
    image = np.zeros((h, w, 3), numba.float32)

    directions = np.zeros((h, w, sample_count, 3), numba.float32)
    weights = np.zeros((h, w, sample_count), numba.float32)
    total_weights = np.zeros((h, w), numba.float32)

    for i in range(h):
        for j in range(w):
            _samples: np.ndarray = stratifiedSample(sample_count)
            samples = np.zeros(sample_count, numba.float32)
            _weights: np.ndarray = np.zeros(sample_count, numba.float32)
            totalWeight = 0
            for k in range(sample_count):
                samples[k] = (_samples[k] - [0.5, 0.5]) * filterWidth
                _weights[k] = gaussian_2D(samples[k][0], samples[k][1], filterWidth)
                totalWeight += _weights[k]

            for k in range(sample_count):
                ray = camera.generateRay(j, i, samples[k])
                directions[i][j][k] = ray.direction
                weights[i][j][k] = _weights[k]
                total_weights[i][j] = totalWeight

    results = intersect(scene_triangles, camera.position, directions)
    for i in results.shape[0]:
        for j in results.shape[1]:
            if results[i][j][0] != np.inf:
                image[i][j] = [1, 1, 1]

    return image


@numba.jit(nopython=True, parallel=True)
def intersect(triangles, origin, directions):
    #print()
    #print()
    size = directions.shape[0:3]
    results = np.full(size, np.inf, np.float32)
    for tri_index in numba.prange(len(triangles)):
        vertices = triangles[tri_index]
        e1 = vertices[1] - vertices[0]
        e2 = vertices[2] - vertices[0]

        for i in numba.prange(size[0]):
            for j in numba.prange(size[1]):
                for s in numba.prange(size[2]):
                    #print('i =', i)
                    r = (origin - vertices[0])
                    mat = np.vstack((e1, e2, -directions[i][j][s]))
                    #print(mat)
                    #print()

                    if np.linalg.cond(mat) < ep:
                        inv = np.ascontiguousarray(np.linalg.inv(mat))
                        u, v, t = r @ inv
                        if 0 <= u <= 1 and 0 <= v <= 1 and t >= 0 and t < results[i][j][s]:
                            results[i][j][s] = t

                        #print(u, v, t)
                        #print()
                        #print()

    #print()
    #print()
    return results


ep = 1 / sys.float_info.epsilon