'''Setup: basic config for visualization of experimental results.'''

## External modules.
from matplotlib.pyplot import axis
from numpy import array


###############################################################################


## Directory names.
img_dir = "img"
results_dir = "results"

## Image parameters.
my_fontsize = "xx-large"
my_ext = "pdf"


## A handy routine for saving just a legend.
def export_legend(legend, filename="legend.pdf", expand=[-5,-5,5,5]):
    '''
    Save just the legend.
    Source for this: https://stackoverflow.com/a/47749903
    '''
    fig = legend.figure
    fig.canvas.draw()
    axis('off')
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)
    return None


###############################################################################
