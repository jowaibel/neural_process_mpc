import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager


def matplotlib_set_publication():
    """Serif + LaTeX publication styling for the paper figures. Returns
    (font_size, legend_size)."""
    render_fine = True
    fsize = 10   # general font
    tsize = 10   # legend font
    tdir = 'in'  # tick direction
    major = 5.0  # major tick length
    minor = 3.0  # minor tick length
    lwidth = 0.8  # frame line width

    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'

    if render_fine:
        plt.rcParams['text.usetex'] = True  # best but slow; needs a LaTeX install
        plt.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'
    else:
        cmfont = font_manager.FontProperties(fname=matplotlib.get_data_path() + '/fonts/ttf/cmr10.ttf')
        plt.rcParams['font.serif'] = cmfont.get_name()
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['axes.formatter.use_mathtext'] = True

    plt.rcParams['font.size'] = fsize
    plt.rcParams['legend.fontsize'] = tsize
    plt.rcParams['xtick.direction'] = tdir
    plt.rcParams['ytick.direction'] = tdir
    plt.rcParams['xtick.major.size'] = major
    plt.rcParams['xtick.minor.size'] = minor
    plt.rcParams['ytick.major.size'] = major
    plt.rcParams['ytick.minor.size'] = minor
    plt.rcParams['axes.linewidth'] = lwidth

    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.borderaxespad'] = 0
    plt.rcParams['legend.handletextpad'] = 0.4

    return fsize, tsize
