"""Plotting style shared by the figure scripts."""
import matplotlib as mpl

mpl.use('Agg')     # headless backend: the same figures on every platform

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
GREY, INK, INK2 = '#8a8984', '#0b0b0b', '#52514e'
SEQ = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']   # blue ramp, light -> dark

mpl.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 1.8, 'legend.frameon': False,
    'savefig.dpi': 200, 'savefig.bbox': 'tight', 'font.family': 'DejaVu Sans',
})
