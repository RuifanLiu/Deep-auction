from routing_model.dep import MPL_ENABLED, matplotlib

def setup_axes_layout(fig, size, aspect_ratio = 4/3):
    r"""
    :param size:         Total number of plots
    :param aspect_ratio: Ratio between width and height of the figure
        Default = 4/3
    """
    if aspect_ratio == 0:
        w,h = 1,size
    elif aspect_ratio >= size:
        w,h = size,1
    else:
        h = (size/aspect_ratio)**0.5
        w = aspect_ratio*h
        w,h = round(w), round(h)
        if w * h < size:
            w += 1
        if w * h < size:
            h += 1
    return fig.subplots(h, w, squeeze = False)


def plot_customers(ax, customers, detailed = False):
    r"""
    :param ax:        Axes object to plot to
    :param customers: :math:`L_c \times D_c` tensor containing customers' features
    :param detailed:  Toggle printing features of each customer next to its node
        Default = False
    :type ax:         matplotlib.pyplot.Axes
    :type customers:  torch.Tensor(dtype = torch.float)
    :type detailed:   bool
    """
    if not MPL_ENABLED:
        raise ImportError("Cannot use plot utils without matplotlib")

    ax.axis('equal')
    ax.set_axis_off()
    if detailed:
        ax.set_title("{} customers".format(customers.size(0)-1))

    maxdem = customers[:,2].max().item()
    mindem = customers[:,2].min().item()
    s = [20+280*(dem-mindem)/(maxdem-mindem) for dem in customers[1:,2].tolist()]
    cmap = matplotlib.cm.autumn

    if customers.size(1) > 3: # TW
        c = customers[1:,4].tolist()
        cnorm = matplotlib.colors.Normalize(0, customers[0,4].item())
    else:
        c = [1.0 for _ in customers[1:]]
        cnorm = matplotlib.colors.Normalize(0, 1)
    
    ax.scatter(*zip(*customers[0:1,:2].tolist()), 200, 'g', 'd')
    ax.scatter(*zip(*customers[1:,:2].tolist()), s, c, 'o', cmap, cnorm)

    for j, cust in enumerate( customers.tolist() ):
        ax.text(cust[0],cust[1],str(j), fontsize = 10,
                horizontalalignment = 'center', verticalalignment = 'center')
        if detailed:
            s = "_{" + str(j) + "}"
            desc = '\n'.join(r"${}{} = {:.2f}$".format(f,s,v) for f,v in zip("xyqelda", cust))
            ax.text(cust[0], cust[1], desc, horizontalalignment = 'left', verticalalignment = 'center')
    return ax


def plot_routes(ax, customers, routes):
    r"""
    :param ax:        Axes object to plot to
    :param customers: :math:`L_c \times D_c` tensor containing customers' features
    :param routes:    Sequence of customers served for every vehicle
    :type ax:         matplotlib.pyplot.Axes
    :type customers:  torch.Tensor(dtype = torch.float)
    :type routes:     list(list(int))
    """
    if not MPL_ENABLED:
        raise ImportError("Cannot use plot utils without matplotlib")

    cmap = matplotlib.cm.tab10
    for i,route in enumerate(routes):
        px,py = customers[0,:2].tolist()
        for j in route[:-1]:
            x,y = customers[j,:2].tolist()
            ax.plot([px,x], [py,y], color = cmap(i), zorder = -1)
            px,py = x,y
        x,y = customers[route[-1],:2].tolist()
        ax.plot([px,x], [py,y], color = cmap(i), zorder = -1, linestyle = '--')
    return ax


def plot_actions(ax, customers, actions, veh_count):
    pxs,pys = customers[0, :2].unsqueeze(1).expand(-1, veh_count).tolist()
    cmap = matplotlib.cm.tab10
    for i,j in actions:
        x,y = customers[j,:2].tolist()
        dashed = '--' if j == 0 else '-'
        ax.plot([pxs[i], x], [pys[i], y], color = cmap(i), zorder = -1, linestyle = dashed)
        pxs[i], pys[i] = x,y
    return ax
