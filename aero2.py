import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(0,1.8,181)

a=0.6
b=0.1
def y(x,a,b):
    y1upper=((b**2)-(b**2/a**2)*(x-a)**2)**0.5
    y2upper=x**3*(b/(4*a**3))-x**2*(3*b/(2*a**2))+x*(9*b/(4*a))
    y1lower=-(4*b/(9*a**2))*(3-(4*x/(3*a)))*(x**2)
    y2lower=-b+(4*b/(3*a**2))*((x-(3*a/2))**2)-(16*b/(27*a**3))*((x-(3*a/2))**3)
    yUpper=np.hstack((y1upper[0:60],y2upper[60:181]))
    yLower=np.hstack((y1lower[0:90],y2lower[90:181]))
    ##calculating derivatives using finite difference convolution, O(dx^2), I could have done this analytically but there was no rule
    dx=0.01
    kernel=[0.5/dx,0,-0.5/dx]
    dyUpperdx=np.convolve(yUpper,kernel,mode='same')
    dyLowerdx=np.convolve(yLower,kernel,mode='same')
    dyLowerdx[0]=(yLower[1]-yLower[0])/dx
    dyLowerdx[180]=(yLower[180]-yLower[179])/dx
    dyUpperdx[0]=(yUpper[1]-yUpper[0])/dx
    dyUpperdx[180]=(yUpper[180]-yUpper[179])/dx
    return(yUpper,yLower,dyUpperdx,dyLowerdx)
yUpper,yLower,_,_=y(x,a,b)
print(x)