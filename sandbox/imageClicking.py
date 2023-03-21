import matplotlib as mpl

# save points clicked on an image
class ClickPoints:
    def __init__(self, ax, N, callback=None):
        self.ax = ax
        self.N = N
        self.callback = callback
        self.cid = ax.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.inaxes != self.ax: return
        if event.button != 1: return
        if len(self.ax.lines) >= self.N: return
        self.ax.plot(event.xdata, event.ydata, 'ro')
        self.ax.figure.canvas.draw()
        if self.callback is not None:
            self.callback(event)

# load and display an image
def imshow(img, ax=None, title=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(img, **kwargs)
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    return ax

