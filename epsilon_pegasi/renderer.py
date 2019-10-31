import numpy as np
import random
from epsilon_pegasi.helpers import enforced_dataclass
from epsilon_pegasi.camera import Camera
from epsilon_pegasi.shapes import Scene
from epsilon_pegasi.base_classes import Ray, Color3

typing_replaces = {np.ndarray: lambda x: np.array(x, dtype=float)}


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


@enforced_dataclass(replaces=typing_replaces)
class Renderer:
    options: RenderOptions
    camera: Camera
    scene: Scene

    def stratifiedSample(self, samples: int) -> np.ndarray:
        size = np.sqrt(samples)
        points = [[0,0] for i in range(samples)]
        
        for i in range(size):
            for j in range(size):
                offset = np.array([i, j])
                points[i * size + j] = (offset + [random.uniform(0, 1), random.uniform(0, 1)]) / size

        return points

    def gamma(self, color: Color3, value: float) -> Color3:
        inverseGamma = 1 / value

        return Color3(color.r ** inverseGamma, color.g ** inverseGamma, color.b ** inverseGamma)

    def exposure(self, color: Color3, value: float) -> Color3:
        power = 2 ** value

        return Color3(color.r * power, color.g * power, color.b * power)

    def saturate(self, x: float)
        return np.clip(x, 0, 1)

    def trace(self, ray: Ray, depth: int) -> Color3:
        intersection = scene.intersects(ray)

        if intersection.hit:
            return Color3(1.0, 1.0, 1.0)

        return Color(0, 0, 0)

    def doge_render(self) -> Image3:
        result = np.zeros((self.camera.width,self.camera.height,3))
        for i in range(self.camera.width):
            for j in range(self.camera.height):
                ray = self.camera.generateRay(i, j)
                intersection= self.scene.intersects(ray)
                result[i][j] = np.where(intersection.hit, (1, 1 ,1),(0,0,0))

        return result

    def willerBrener_render(self) -> Image3:
        for i in range(self.camera.width - 1):
            for j in range(self.camera.height - 1):
                samples: np.ndarray = stratifiedSample(self.options.cameraSamples)

                color = Color3(0,0,0)
                totalWeight = 0

                for k in range(self.options.camerasamples):
                    sample = (samples[k] - Point(0.5, 0.5)) * filterWidth
                    ray = camera.generateRay(i, j, sample)
                    weigth = gaussian2D(sample, filterWidth)

                    color += trace(ray, 0) * weigth
                    totalWeight += weight

                color /= totalWeight
                image[i][j] = self.saturate(self.gamma(self.exposure(color, 0.0), 2.2)) * 255