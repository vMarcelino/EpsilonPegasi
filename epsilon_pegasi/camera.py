import numpy as np
import numba
from epsilon_pegasi.helpers import enforced_dataclass
from epsilon_pegasi.base_classes import Ray

typing_replaces = {np.ndarray: lambda x: np.array(x, dtype=float)}


@numba.jit
def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


@enforced_dataclass(replaces=typing_replaces)
class Camera:
    fieldOfView: float
    width: float
    height: float
    position: np.ndarray = np.array([0, 0, 0], dtype=float)
    worldMatrix: np.ndarray = np.zeros([4, 4])

    def aspectRatio(self) -> float:
        return self.width / self.height

    def lookAt(self, position: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0, 0, 1],
                                                                                         dtype=float)) -> None:
        self.position = position
        reverse_forward: np.ndarray = normalize(self.position - target)
        right: np.ndarray = normalize(np.cross(up, reverse_forward))
        corrected_up: np.ndarray = np.cross(reverse_forward, right)

        # self.worldMatrix = [right, corrected_up, reverse_forward, self.position]
        rotation_matrix = [reverse_forward, right, corrected_up, [0, 0, 0]]  # ?
        # [ ux, uy, uz ]
        # [ vx, vy, vz ]
        # [ wx, wy, wz ]
        # [ px, py, pz ]

        rotation_matrix = np.concatenate((rotation_matrix, np.transpose([[0, 0, 0, 1.0]])), axis=1)
        # [ ux, uy, uz ] [0]
        # [ vx, vy, vz ] [0]
        # [ wx, wy, wz ] [0]
        # [ px, py, pz ] [1]
        print(rotation_matrix)
        print()

        translation_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1], self.position]
        translation_matrix = np.concatenate((translation_matrix, np.transpose([[0, 0, 0, 1.0]])), axis=1)
        print(translation_matrix)
        print()

        self.worldMatrix = rotation_matrix @ translation_matrix
        print(self.worldMatrix)
        print()

    def generateRay(self, x: float, y: float, sample: np.ndarray) -> Ray:
        scale = np.tan(self.fieldOfView / 2)

        pixel = [0, 0, 0, 1]

        pixel[0] = -1.0
        pixel[1] = (2.0 * (x + sample[0] + 0.5) / self.width - 1.0) * scale * self.aspectRatio()
        pixel[2] = (1.0 - 2.0 * (y + sample[1] + 0.5) / self.height) * scale

        pixel = pixel @ self.worldMatrix
        #                  [ ux, uy, uz, 0 ]
        #                  [ vx, vy, vz, 0 ]
        #        x         [ wx, wy, wz, 0 ]
        #                  [ tx, ty, tz, 1 ]
        #
        # [ px, py, pz, 1 ]

        #print()
        #print(pixel)
        #pixel = pixel / pixel[3] # no need
        pixel = pixel[0:3]
        #print(pixel)

        #position: np.ndarray = (self.worldMatrix[3][0], self.worldMatrix[3][1], self.worldMatrix[3][2])
        direction: np.ndarray = normalize(pixel - self.position)

        return Ray(self.position, direction)


@numba.jitclass([('fieldOfView', numba.float32), ('width', numba.float32), ('height', numba.float32),
                 ('position', numba.float32[:]), ('worldMatrix', numba.float32[:, :])])
class OptimizedCamera:
    def __init__(self, fieldOfView: float, width: float, height: float):

        self.fieldOfView = fieldOfView
        self.width = width
        self.height = height
        self.position = np.array((0, 0, 0), numba.float32)
        self.worldMatrix = np.zeros((4, 4), numba.float32)

    def aspectRatio(self) -> float:
        return self.width / self.height

    def lookAt(self, position: np.ndarray, target: np.ndarray, up: np.ndarray):
        self.position = position
        reverse_forward: np.ndarray = normalize(self.position - target)
        right: np.ndarray = normalize(np.cross(up, reverse_forward))
        corrected_up: np.ndarray = np.cross(reverse_forward, right)

        rotation_matrix = np.vstack((reverse_forward, right, corrected_up, np.array((0, 0, 0), numba.float32)))
        to_add = np.array([[0, 0, 0, 1.0]], numba.float32)
        to_add_T = np.transpose(to_add)
        rotation_matrix = np.concatenate((rotation_matrix, to_add_T), axis=1)
        print(rotation_matrix)
        print()

        #translation_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1], self.position]
        translation_matrix = np.vstack((np.array((1, 0, 0), numba.float32), np.array(
            (0, 1, 0), numba.float32), np.array((0, 0, 1), numba.float32), self.position))
        to_add = np.array([[0, 0, 0, 1.0]], numba.float32)
        to_add_T = np.transpose(to_add)
        translation_matrix = np.concatenate((translation_matrix, to_add_T), axis=1)
        print(translation_matrix)
        print()

        #self.worldMatrix = rotation_matrix @ translation_matrix
        wm = rotation_matrix @ translation_matrix

        print(self.worldMatrix)
        print()

    def generateRay(self, x: float, y: float, sample: np.ndarray) -> Ray:
        scale = np.tan(self.fieldOfView / 2)

        pixel = [0, 0, 0, 1]

        pixel[0] = -1.0
        pixel[1] = (2.0 * (x + sample[0] + 0.5) / self.width - 1.0) * scale * self.aspectRatio()
        pixel[2] = (1.0 - 2.0 * (y + sample[1] + 0.5) / self.height) * scale

        pixel = pixel @ self.worldMatrix
        #                  [ ux, uy, uz, 0 ]
        #                  [ vx, vy, vz, 0 ]
        #        x         [ wx, wy, wz, 0 ]
        #                  [ tx, ty, tz, 1 ]
        #
        # [ px, py, pz, 1 ]
        pixel = pixel[0:3]
        direction: np.ndarray = normalize(pixel - self.position)

        return Ray(self.position, direction)