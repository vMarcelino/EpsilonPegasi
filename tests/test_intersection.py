import numpy as np

# "python.testing.pytestArgs": [
#     "--cov=epsilon_pegasi"
# ],


def test_intersection_success_negative_origin():
    from epsilon_pegasi import Triangle, Vertex, Scene, Ray, Intersection
    triangle = Triangle(vertices=[Vertex([0, -1, -1]), Vertex([0, 1, -1]), Vertex([0, 0, 1])])
    scene = Scene(shapes=[triangle])
    intersection = scene.intersects(Ray(origin=[-1, 0, 0], direction=[1, 0, 0]))
    expected_answer = Intersection(hit=True, distance=1, object_hit=triangle)
    assert intersection == expected_answer


def test_intersection_success_positive_origin():
    from epsilon_pegasi import Triangle, Vertex, Scene, Ray, Intersection
    triangle = Triangle(vertices=[Vertex([0, -1, -1]), Vertex([0, 1, -1]), Vertex([0, 0, 1])])
    scene = Scene(shapes=[triangle])
    intersection = scene.intersects(Ray(origin=[1, 0, 0], direction=[-1, 0, 0]))
    expected_answer = Intersection(hit=True, distance=1, object_hit=triangle)
    assert intersection == expected_answer


def test_intersection_no_hit():
    from epsilon_pegasi import Triangle, Vertex, Scene, Ray, Intersection
    import math
    triangle = Triangle(vertices=[Vertex([0, -1, -1]), Vertex([0, 1, -1]), Vertex([0, 0, 1])])
    scene = Scene(shapes=[triangle])
    intersection = scene.intersects(Ray(origin=[-1, 0, 0], direction=[0, 1, 0]))
    expected_answer = Intersection(hit=False, distance=math.inf, object_hit=None)
    assert intersection == expected_answer


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath('..'))
    test_intersection_success_negative_origin()
    test_intersection_success_positive_origin()
    test_intersection_no_hit()