import numpy as np

# "python.testing.pytestArgs": [
#     "--cov=epsilon_pegasi"
# ],


def test_intersection():
    print('')
    from epsilon_pegasi import Triangle, Vertex, Scene, Ray
    triangle = Triangle(vertices=[Vertex([0, -1, -1]), Vertex([0, 1, -1]), Vertex([0, 0, 1])])
    scene = Scene(shapes=[triangle])
    intersection = scene.intersects(Ray(origin=[-1, 0, 0], direction=[1, 0, 0]))
    print(intersection)
    assert True


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath('..'))
    test_intersection()