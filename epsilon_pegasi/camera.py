import numpy as np
from epsilon_pegasi.helpers import enforced_dataclass
from epsilon_pegasi.base_classes import Ray

typing_replaces = {np.ndarray: lambda x: np.array(x, dtype=float)}


@enforced_dataclass(replaces=typing_replaces)
class Camera:
    fieldOfView: float
    width: float
    height: float
    position: np.ndarray = np.array([0, 0, 0])
    worldMatrix: np.ndarray = np.zeros([4, 4])

    def aspectRatio(self) -> float:
        return self.width / self.height

    def lookAt(self, position: np.ndarray, target: np.ndarray, up: np.ndarray) -> None:
        self.position = position
        w: np.ndarray = np.linalg.norm(self.position - target)
        u: np.ndarray = np.linalg.norm(up.cross(w))
        v: np.ndarray = w.cross(u)

        self.worldMatrix = np.matrix([u, v, w, self.position])
        # [ ux, uy, uz ]
        # [ vx, vy, vz ]
        # [ wx, wy, wz ]
        # [ px, py, pz ]

        self.worldMatrix = np.concatenate((self.worldMatrix, np.matrix([0, 0, 0, 1]).T), axis=1)
        # [ ux, uy, uz ] [0]
        # [ vx, vy, vz ] [0]
        # [ wx, wy, wz ] [0]
        # [ px, py, pz ] [1]

    def generateRay(self, x: float, y: float
                    #, sample: np.ndarray
                    ) -> Ray:
        scale = np.tan(self.fieldOfView / 2)

        pixel = [0, 0, 0, 1]

        # pixel[0] = (2.0 * (x + sample[0] + 0.5) / self.width - 1.0) * scale * self.aspectRatio()
        # pixel[1] = (1.0 - 2.0 * (y + sample[1] + 0.5) / self.height) * scale

        pixel[0] = scale * (2 * x + 1 - self.width) / self.height
        pixel[1] = scale * (self.height - (2 * y + 1)) / self.height
        pixel[2] = -1.0

        pixel = self.worldMatrix * np.matrix(pixel).T
        pixel = pixel.A1[0:3]

        #position: np.ndarray = (self.worldMatrix[3][0], self.worldMatrix[3][1], self.worldMatrix[3][2])
        direction: np.ndarray = np.linalg.norm(pixel - self.position)

        return Ray(self.position, direction)