import numpy as np
import matplotlib.pyplot as plt


def vis(A):
    plt.imshow(A, cmap='gray')
    if A.dtype != 'bool':
        plt.colorbar()
    plt.show()


def showimg(i, figsize=(8,8), cmap='gray', title=None, cb=False, vmin=None, vmax=None):
    if vmin is None:
        vmin = np.amin(i)
    if vmax is None:
        vmax = np.amax(i)
    fig, ax = plt.subplots(1,1,figsize=figsize)
    ax.imshow(i, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    if title:
        plt.title(title)
    if cb:
        plt.colorbar()
    plt.show()

def showcomp(imgs, figsize=(14,6), titles=None, vmin=None, vmax=None, horizontal=True):

    if not isinstance(vmin, list):
        vmin = [vmin,]*len(imgs)

    if not isinstance(vmax, list):
        vmax = [vmax,]*len(imgs)

    if horizontal:
        fig, axs = plt.subplots(1,len(imgs), figsize=figsize)
    else:
        fig, axs = plt.subplots(len(imgs),1, figsize=figsize)

    for i in range(len(imgs)):
        axs[i].imshow(imgs[i], cmap='gray', vmin=vmin[i], vmax=vmax[i])
        if titles is not None:
            if isinstance(titles, list):
                axs[i].set_title(titles[i])
        axs[i].set_axis_off()

    plt.show(fig)


def vis_grid(As, n_cols, figsize_weight=[3,2], cmap='gray', filepath=None):
    n_rows = (len(As) + len(As)%n_cols) // n_cols

    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(n_cols*figsize_weight[0], n_rows*figsize_weight[1])
    )

    for i in range(n_rows):
        for j in range(n_cols):
            axs[i,j].imshow(As[j*n_cols+i], cmap=cmap)

            if i==(n_rows-1) and j==0:
                axs[i,j].axis('on')
            else:
                axs[i,j].axis('off')

    if filepath:
        fig.savefig(filepath, dpi=300)
    plt.show(fig)


def gplot(G, im=None, c=None, cmap='hot', figsize=(8,8), savefile=None, vmin=0, vmax=20, cb=False):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if im is not None:
        ax.imshow(im)

    p = ax.scatter(G[:,1], G[:,0], c=c, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_ylim(504,0)
    ax.set_xlim(0,504)
    ax.set_aspect('equal', 'box')
    # ax.set_facecolor((0.5,0.5,0.5))

    if cb:
        plt.colorbar(p)

    if savefile:
        fig.savefig(savefile)
        plt.close('all')
    else:
        plt.show(fig)
