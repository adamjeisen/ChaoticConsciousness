import matplotlib.pyplot as plt
import numpy as np

def barplot_annotate_brackets(num1, num2, p, center, height, it=0, ax=None, ylim=None, scale_by_height=False, yerr=None, dh=.05, barh=.05, gap=None, fs=None, maxasterix=None, below=False):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """
    
    data = gen_data(p)
    
    if ax is None:
        ax = plt.gca()
    
    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'
#     print(np.log10(height))

    if gap is None:
        gap = dh

    lx = center[num1]
    if below:
        ly = np.min([height[num1], height[num2]])
    else:
        ly = np.max([height[num1], height[num2]])
    rx = center[num2]
    if below:
        ry = np.min([height[num1], height[num2]])
    else:
        ry = np.max([height[num1], height[num2]])

    if yerr is not None:
        if ly >= 0:
            ly += yerr[num1]
        else:
            ly -= yerr[num1]
        if ry >= 0:
            ry += yerr[num2]
        else:
            ry -= yerr[num2]

    if scale_by_height:
        ax_y0, ax_y1 = np.min(np.array(height) + np.array(yerr)), np.max(np.array(height) + np.array(yerr))
    else:
        if ylim is None:
            ax_y0, ax_y1 = ax.get_ylim()
        else:
            ax_y0, ax_y1 = ylim
    dh *= (ax_y1 - ax_y0)
    gap *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)
#     barh *= (np.max(height) - np.min(height))
    
    if ax.get_yscale() == 'log':
        mult = np.power(4, it/2)
    else:
        mult = 1
    
    if below:
        y = min(ly, ry) - dh - mult*it*gap 
    else:
        y = max(ly, ry) + dh + mult*it*gap
    
   

    barx = [lx, lx, rx, rx]
    if below:
        bary = [y, y-barh, y-barh, y]
        mid = ((lx+rx)/2, y-barh)
    else:
        bary = [y, y+barh, y+barh, y]
        mid = ((lx+rx)/2, y+barh)

    ax.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    ax.text(*mid, text, **kwargs)

def gen_data(p):
    if p == 0 or p < 1e-15:
        data = 'p < 1e-15'
    elif p <= 0.05 and p >= 1e-4:
        data = p
    else:
        data = f"p < 1e-{int(-np.log10(p))}"
    
    return data