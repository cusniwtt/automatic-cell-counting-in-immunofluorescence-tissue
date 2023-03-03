from rich.progress import track
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
import skimage.io as io

def create_stack(ref, label):
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Image(z=ref), row=1, col=1)
    fig.add_trace(go.Image(z=label), row=1, col=2)
    return fig

dataset = ['exp29/', 'exp31/', 'exp33/']
mask_path = 'masktest/'
save_path = ['stack/exp29', 'stack/exp31', 'stack/exp33']

init = 0
for p in dataset:
    filenames = os.listdir(p)
    filenames.sort()
    filenames = filenames[1:]
    paths = [p + str(x) for x in filenames]
    paths = [x for x in paths if x.endswith('.png')]

    mask = [str(mask_path) + str(x) for x in filenames]
    mask = [x for x in mask if x.endswith('.png')]

    zip_paths = list(zip(paths, mask))

    for file in track(zip_paths):
        label_img = io.imread(file[0])
        ref = io.imread(file[1])
        stack_fig = create_stack(ref, label_img)
        save_file = save_path[init] + file[0][5:-4] + '_labeled.png'
        stack_fig.write_image(save_file)
    init += 1
