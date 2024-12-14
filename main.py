import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange

class MandelbrotPlotter:
    def __init__(self, xmin, xmax, ymin, ymax, width, height, max_iter):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.zoom_history = []
        self.fig, self.ax = plt.subplots()
        n3 = MandelbrotPlotter.mandelbrot_set(
            self.xmin, self.xmax, self.ymin, self.ymax,
            self.width, self.height, self.max_iter, MandelbrotPlotter.mandelbrot
        )
        self.im = self.ax.imshow(
            n3.T, 
            extent=[self.xmin, self.xmax, self.ymin, self.ymax], 
            cmap='hot'
        )
        self.ax.set_title(f'Mandelbrot Set\nxmin={self.xmin}, xmax={self.xmax}, ymin={self.ymin}, ymax={self.ymax}')
        self.colorbar = self.fig.colorbar(self.im, ax=self.ax)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        plt.show()

    @staticmethod
    @jit(nopython=True)
    def mandelbrot(c, max_iter):
        z = c
        for n in range(max_iter):
            if abs(z) > 2:
                return n
            z = z*z + c
        return max_iter

    @staticmethod
    @jit(nopython=True, parallel=True)
    def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter, mandelbrot_func):
        r1 = np.linspace(xmin, xmax, width)
        r2 = np.linspace(ymin, ymax, height)
        n3 = np.empty((width, height), dtype=np.int32)
        for i in prange(width):
            for j in range(height):
                n3[i, j] = mandelbrot_func(r1[i] + 1j*r2[j], max_iter)
        return n3

    def plot_mandelbrot(self):
        n3 = MandelbrotPlotter.mandelbrot_set(
            self.xmin, self.xmax, self.ymin, self.ymax,
            self.width, self.height, self.max_iter, MandelbrotPlotter.mandelbrot
        )
        self.im.set_data(n3.T)
        self.im.set_extent([self.xmin, self.xmax, self.ymin, self.ymax])
        self.ax.set_title(f'Mandelbrot Set\nxmin={self.xmin}, xmax={self.xmax}, ymin={self.ymin}, ymax={self.ymax}')
        self.colorbar.update_normal(self.im)
        plt.draw()

    def on_click(self, event):
        if event.inaxes:
            self.zoom_history.append((self.xmin, self.xmax, self.ymin, self.ymax, self.max_iter))
            x, y = event.xdata, event.ydata
            zoom_factor = 0.5
            x_range = (self.xmax - self.xmin) * zoom_factor
            y_range = (self.ymax - self.ymin) * zoom_factor
            self.xmin = x - x_range / 2
            self.xmax = x + x_range / 2
            self.ymin = y - y_range / 2
            self.ymax = y + y_range / 2
            self.max_iter = int(self.max_iter * 1.5)
            self.plot_mandelbrot()

    def on_key_press(self, event):
        if event.key == 'b' and self.zoom_history:
            self.xmin, self.xmax, self.ymin, self.ymax, self.max_iter = self.zoom_history.pop()
            self.plot_mandelbrot()

xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
width, height, max_iter = 1000, 1000, 256
MandelbrotPlotter(xmin, xmax, ymin, ymax, width, height, max_iter)