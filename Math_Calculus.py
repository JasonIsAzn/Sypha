import sympy as sp
import numpy as np
from tkinter import *
from sympy.parsing.sympy_parser import parse_expr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

######  Limits ######
class LimitData:
    def __init__(self, eqn, var, val, direct):
        self.eqn = eqn
        self.var = var
        self.val = val
        self.direct = direct


def limit_class(str_info):
    """
    Create a LimitData Class with given parameters

    Parameters:
        str_info (tuple): string tuple (Equation, variable, value, direction)

    Returns:
        my_limit (LimitData): LimitData class of the given limit problem
    """

    eqn, var, val, direct = str_info[0], str_info[1], str_info[2], str_info[3]

    eqn_ = parse_expr(eqn)
    var_ = sp.symbols(var)
    val_ = sp.oo if val == "oo" else -sp.oo if val == "-oo" else int(val)

    MyLimit = LimitData(eqn_, var_, val_, direct)
    return MyLimit


def limit_find(MyLimit):
    """
    Limit of function with plotting info

    Parameters:
        MyLimit (LimitData): Class of the Limit Problem

    Returns:
        lim (int): The answer to the limit
        plot_info (tuple): Info about plot (plot_type, xrange, yrange, lim)
    """

    eqn, var, val, direct = MyLimit.eqn, MyLimit.var, MyLimit.val, MyLimit.direct
    inf = [sp.oo, -sp.oo, sp.zoo, -sp.zoo]

    lim = sp.limit(eqn, var, val, direct)

    interval = 3
    if type(val) == int:
        if lim in inf:
            plot_type = "vert"
            xrange = [-10, 10]
            yrange = [-10, 10]
        else:
            plot_type = "point"
            xrange = [val - interval, val + interval]
            yrange = [lim - interval, lim + interval]
    else:
        if lim in inf:
            plot_type = "graph_only"
            xrange = [-10, 10]
            yrange = [-10, 10]
        else:
            plot_type = "hort"
            xrange = [-interval, interval]
            yrange = [-interval, interval]

    xvals = np.arange(xrange[0], xrange[1], 0.1)
    yvals = np.array([eqn.subs(var, i) for i in xvals])
    maxi = max(yvals)
    mini = min(yvals)

    if maxi > 10 ** 10:
        pass
    else:
        yrange[1] = maxi + interval
    if mini < -10 ** 10:
        pass
    else:
        yrange[0] = mini - interval

    plot_info = (plot_type, xrange, yrange, lim)
    return lim, plot_info


def limit_plot(MyLimit, plot_info):
    """
    plotting equation for limits

    Parameters:
        MyLimit (LimitData): LimitData class of given limit problem
        plot_info (tuple): Info about the plot  (plot_type, xrange, yrange)

    Returns:
        fig (matplotlib fig): fig of the plot
    """
    eqn, var, val, direct = MyLimit.eqn, MyLimit.var, MyLimit.val, MyLimit.direct
    plot_type, xrange, yrange, ans = plot_info
    fig = plt.figure()

    xvals = np.arange(xrange[0], xrange[1], 0.1)
    yvals = np.array([eqn.subs(var, i) for i in xvals])

    plt.plot(xvals, yvals, 'b')

    if plot_type == "vert":
        plt.axvline(x=val, color='r', linestyle='-')
    elif plot_type == "hort":
        plt.axhline(y=ans, color='r', linestyle="-")
    elif plot_type == "point":
        plt.plot(val, ans, 'r*')

    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.xlim(int(xrange[0]), int(xrange[1]))
    plt.ylim(int(yrange[0]), int(yrange[1]))
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.title("Function = " + str(eqn))
    plt.grid(True)

    return fig


def plot(fig):
    # ALPHA GUI
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack()
    toolbar = NavigationToolbar2Tk(canvas, window)
    toolbar.update()
    canvas.get_tk_widget().pack()


def limit_test():
    # TEST: Data is given in string format should output the an answer and a fig corresponding to limit
    data_pre = ("1/x", 'x', '0', '+-')
    data = limit_class(data_pre)
    ans, plot_info = limit_find(data)
    fig = limit_plot(data, plot_info)
    print("My Limit:", data.eqn)
    print("Ans: " + str(ans))
    print("Plot_Info: " + str(plot_info))
    return ans, fig


show_plot = False
try:
    ans, fig = limit_test()
    show_plot = True
except ValueError:
    print("The limit does not exist. ")
    pass

if show_plot:
    window = Tk()
    window.title("Plotting")
    window.geometry("500x500")
    plot_button = Button(master=window,
                         command=plot(fig),
                         height=2,
                         width=10,
                         text="Plot")
    plot_button.pack()
    window.mainloop()


##### Derivatives ######

class DerivativeData:
    def __init__(self, eqn, order, var1, var2):
        self.eqn = eqn
        self.order = order
        self.var1 = var1
        self.var2 = var2


def derivative_data(str_info):
    """
    Create a DerivativeData Class with given parameters

    Parameters:
        str_info (tuple): string tuple (eqn, order, var1, var2)

    Returns:
        my_der (DerivativeData): LimitData class of the given limit problem
    """
    eqn, order, var1, var2 = str_info[0], str_info[1], str_info[2], str_info[3]

    eqn_ = parse_expr(eqn)
    order_ = int(order)
    var1_ = sp.symbols(var1)
    var2_ = False if var2 == "False" else sp.symbols(var2)

    my_der = DerivativeData(eqn_, order_, var1_, var2_)
    return my_der


def derivative_find(my_der):
    """
    Find Derivative of Either Regular or Implicit

    Parameters:
        my_der (DerivativeData): Class of the Derivative problem

    Returns:
        ans (list of sympy_eqn): List of Derivative of the Equation

    """
    eqn, order, var1, var2 = my_der.eqn, my_der.order, my_der.var1, my_der.var2

    ans = []  # list with n order derivatives 
    if not isinstance(var2, sp.core.symbol.Symbol):  # explicit
        for num in range(1, order + 1):
            ans.append(eqn.diff(var1, num))
    else:  # implicit
        for num in range(1, order + 1):
             ans.append(sp.idiff(eqn, var2, var1, n=num))

    return ans


def derivative_plot(my_der, ans):
    """
    Plot the Original Equation and Derivative Equation

    Parameters:
        my_der (DerivativeData): class of the Derivative problem
        ans (list of sympy_eqn): List of Derivative of the Equation

    Returns:
        fig (matplotlib fig): figure of the plot

    """

    eqn, order, var1, var2 = my_der.eqn, my_der.order, my_der.var1, my_der.var2

    if isinstance(var2, sp.core.symbol.Symbol): # If Implicit Function, Don't Plot
        return False

    interval = 5
    xvals = np.arange(-interval, interval, .1)
    org_yvals = np.array([eqn.subs(var1, i) for i in xvals])
    der_yvals = np.array([ans[-1].subs(var1, i) for i in xvals])

    fig, ax = plt.subplots()
    plt.plot(xvals, org_yvals, 'b', label="f(" + str(var1) + ")")
    plt.plot(xvals, der_yvals, 'r', label="f" + order * "'" + "(" + str(var1) + ")")

    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    ax.set_title("Function: " + str(eqn))
    ax.set_xlabel("X-Axis")
    ax.set_ylabel("Y-Axis")
    plt.grid(True)
    ax.legend()
    ax.relim()
    ax.autoscale_view()

    plt.show()
    return fig


def plot(fig):
    # ALPHA GUI
    if not fig:
        pass
    else:
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack()
        toolbar = NavigationToolbar2Tk(canvas, window)
        toolbar.update()
        canvas.get_tk_widget().pack()


def derivative_test():
    # TEST: Data is given in string format should output the an answer and a fig corresponding to limit
    data_pre = ("x**2 + x*y", '3', 'x', 'False')
    data = derivative_data(data_pre)
    ans = derivative_find(data)
    fig = derivative_plot(data, ans)
    
    print("f(x)= " + str(data.eqn))
    for i, der in enumerate(ans):
        print("f" + (i + 1) * "'" + "(x)= " + str(der))
    return ans, fig


ans, fig = derivative_test()
window = Tk()
window.title("Plotting")
window.geometry("500x500")
plot_button = Button(master=window,
                     command=plot(fig),
                     height=2,
                     width=10,
                     text="Plot")
plot_button.pack()
window.mainloop()


