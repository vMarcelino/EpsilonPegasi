from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass
from epsilon_pegasi.base_classes import BSDF, Vertex, Ray, ShaderGlobals
from epsilon_pegasi.helpers import enforced_dataclass
import math

typing_replaces = {np.ndarray: lambda x: np.array(x, dtype=float)}


class Shape:
    bsdf: BSDF

    @abstractmethod
    def intersects(self, ray: Ray) -> Intersection:
        raise NotImplementedError

    @abstractmethod
    def calculateShaderGlobals(self, ray: Ray, intersection: Intersection) -> ShaderGlobals:
        raise NotImplementedError

    @abstractmethod
    def surfaceArea(self) -> float:
        raise NotImplementedError


@enforced_dataclass(exceptions=[Shape], blacklist=True)
class Intersection:
    hit: bool
    distance: float = None
    object_hit: Shape = None  # pylint: disable=used-before-assignment


@dataclass
class Scene:
    shapes: List[Shape]

    def intersects(self, ray: Ray) -> Intersection:
        result = Intersection(hit=False, distance=math.inf)
        for intersection in [x.intersects(ray) for x in self.shapes]:
            if intersection.hit and intersection.distance < result.distance:
                result = intersection

        return result


@dataclass
class Triangle(Shape):
    vertices: List[Vertex]

    def _get_intersection_distance(self, origin, direction):
        EPSILON = 0.0000001
        edge_1 = self.vertices[1].position - self.vertices[0].position
        edge_2 = self.vertices[2].position - self.vertices[0].position
        part_of_determinant_d_e2 = np.cross(direction, edge_2)
        determinant = np.dot(part_of_determinant_d_e2, edge_1)

        is_ray_parallel = lambda d: -EPSILON < d < EPSILON
        is_ray_parallel = np.vectorize(is_ray_parallel)
        np.where(is_ray_parallel(determinant), np.inf)

        is_ray_parallel = -EPSILON < determinant < EPSILON
        if is_ray_parallel:
            return Intersection(hit=False)

        Tau = ray.origin - self.vertices[0].position
        u = Tau.dot(part_of_determinant_d_e2) / determinant
        if not (0 <= u <= 1):
            return Intersection(hit=False)

        part_of_determinant_e1_tau = np.cross(Tau, edge_1)
        v = ray.direction.dot(part_of_determinant_e1_tau) / determinant
        if v < 0 or u + v > 1:
            return Intersection(hit=False)

        scalar_distance = edge_2.dot(part_of_determinant_e1_tau) / determinant

        if EPSILON < scalar_distance < (1 / EPSILON):
            return Intersection(hit=True, distance=scalar_distance, object_hit=self)
        else:
            return Intersection(hit=False)

    def intersects(self, ray: Ray) -> Intersection:
        EPSILON = 0.0000001
        edge_1 = self.vertices[1].position - self.vertices[0].position
        edge_2 = self.vertices[2].position - self.vertices[0].position

        # h = rayVector.crossProduct(edge2);
        part_of_determinant_d_e2 = np.cross(ray.direction, edge_2)

        # a = edge1.dotProduct(h);
        determinant: float = np.dot(part_of_determinant_d_e2, edge_1)

        is_ray_parallel = -EPSILON < determinant < EPSILON  # det = 0: matrix is not inversible
        if is_ray_parallel:
            return Intersection(hit=False)

        # f = 1.0/a;
        #f = 1 / cosine_between_intersectionNormal_and_edge

        # s = rayOrigin - vertex0;
        Tau = ray.origin - self.vertices[0].position

        # u = f * s.dotProduct(h);
        u = Tau.dot(part_of_determinant_d_e2) / determinant

        # if (u < 0.0 || u > 1.0)
        #     return false;
        if not (0 <= u <= 1):
            return Intersection(hit=False)

        # q = s.crossProduct(edge1);
        part_of_determinant_e1_tau = np.cross(Tau, edge_1)

        # v = f * rayVector.dotProduct(q);
        v = ray.direction.dot(part_of_determinant_e1_tau) / determinant

        # if (v < 0.0 || u + v > 1.0)
        #     return false;
        if v < 0 or u + v > 1:
            return Intersection(hit=False)

        # // At this stage we can compute t to find out where the intersection point is on the line.
        # float t = f * edge2.dotProduct(q);
        scalar_distance = edge_2.dot(part_of_determinant_e1_tau) / determinant

        # if (t > EPSILON && t < 1/EPSILON) // ray intersection
        # {
        #     outIntersectionPoint = rayOrigin + rayVector * t;
        #     return true;
        # }
        if EPSILON < scalar_distance < (1 / EPSILON):
            #intersection_point: np.ndarray = ray.origin + ray.direction * scalar_distance
            return Intersection(hit=True, distance=scalar_distance, object_hit=self)

        # else // This means that there is a line intersection but not a ray intersection.
        #     return false;
        else:
            return Intersection(hit=False)

    def surfaceArea(self) -> float:
        edge_1 = self.vertices[1].position - self.vertices[0].position
        edge_2 = self.vertices[2].position - self.vertices[0].position

        vectorial_product = np.cross(edge_1, edge_2)
        return np.sqrt(vectorial_product.dot(vectorial_product))


@enforced_dataclass(replaces=typing_replaces)
class Sphere:
    position: np.ndarray
    radius: float
