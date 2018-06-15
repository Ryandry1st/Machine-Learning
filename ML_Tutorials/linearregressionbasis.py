#program which does a linear regression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random


style.use("fivethirtyeight")


def create_dataset(num, variance, step=2, correlation=False):
    val = 1
    yset = []
    for i in range(num):
        y = val+random.randrange(-variance, variance)
        yset.append(y)
        if correlation and correlation == "pos":
            val+=step
        elif correlation and correlation =="neg":
            val -=step
    xset = [i for i in range(len(yset))]

    return np.array(xset, dtype=np.float64), np.array(yset, dtype=np.float64)

    
def best_fit_slope(xs, ys):
    m = (((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) /
         ((np.mean(xs)**2) - np.mean(xs**2)))
    return m


def linearfit(xs, yx):
    m = best_fit_slope(xs, ys)
    b = np.mean(ys) - m*np.mean(xs)

    return m, b


# determine how good a fit the line is
#look for r^2 value, r^2 = 1-(SE*regressionline)/(SE mean(y))
def r_squared(reg_line, yvals):
    SEYhat = sum((reg_line-yvals)**2)
    SEYav = sum(([np.mean(yvals) for y in yvals] - yvals)**2)
    #returns r^2
    return 1- SEYhat/SEYav
    


if __name__=='__main__':
    
    xs, ys = create_dataset(40, 10, 2, correlation='pos')
    m , b = linearfit(xs, ys)

    # creates simple y values for xs
    regression_line = [(m*x)+b for x in xs]
    predict_x = 8
    predict_y = m*predict_x+b

    

    print r_squared(regression_line, ys)
    
    plt.scatter(xs, ys)
    plt.scatter(predict_x, predict_y, s=50, color='g')
    plt.plot(xs, regression_line)
    plt.show()

