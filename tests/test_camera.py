import numpy as np
import math

# "python.testing.pytestArgs": [
#     "--cov=epsilon_pegasi"
# ],


def _test_renderer_with_plot():
    from epsilon_pegasi import Triangle, Vertex, Scene, Ray, Intersection, RenderOptions, Renderer, Camera
    triangle = Triangle(vertices=[Vertex([0, -1, -1]), Vertex([0, 1, -1]), Vertex([0, 0, 1])])  # z is up
    #triangle = Triangle(vertices=[Vertex([-1, -1, 0]), Vertex([1, -1, 0]), Vertex([0, 1, 0])]) # y is up
    scene = Scene(shapes=[triangle])
    w, h = 256, 128
    options = RenderOptions(width=w,
                            height=h,
                            maximumDepth=999999,
                            cameraSamples=4,
                            lightSamples=1,
                            diffuseSamples=1,
                            filterWidth=2,
                            gamma=2.2,
                            exposure=1)
    camera = Camera(fieldOfView=90 * math.pi / 180, width=w, height=h)
    camera.lookAt(position=np.array([5, 0, 0], dtype=float), target=np.array([0, 0, 0], dtype=float))  # z is up
    #camera.lookAt(position=np.array([0, 0, 30], dtype=float), target=np.array([0, 0, 0], dtype=float))  # y is up
    r = Renderer(options, camera, scene)
    #img = r.doge_render()
    img = r.willerBrener_render()
    from matplotlib import pyplot as plt
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath('..'))
    _test_renderer_with_plot()