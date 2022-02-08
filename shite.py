import numpy
import sympy

from matplotlib import pyplot
from sympy import init_printing
from sympy.utilities.lambdify import lambdify
from matplotlib.animation import FuncAnimation


#################################################
#################################################
#################################################
#setting initial functions for potential and wavefunction

x,t = sympy.symbols('x t')

initphi = (-sympy.Abs(x)+3 ) * (sympy.exp(-x**2*1000))                      #wavefunction at t=0
fn_init = sympy.lambdify((x) , initphi)
#print(fn_init(0))

initv = x**2 * 4                        #Potential V 
fn_v = sympy.lambdify((x) , initv)
#print(fn_v(0))

#################################################
#################################################
#################################################

#parameters for grid
nx = 301
nt = 10000
dx = 1/(nx-1)
dt = dx * 1e-4
x = numpy.linspace(-1,1,nx)
t = 0

#################################################
#################################################
#################################################

#declaring arrays, u is t+1, un is t, v is potential
u = numpy.asarray([fn_init(xi) for xi in x], dtype=complex)
un = numpy.empty(nx, dtype=complex)
v = numpy.asarray([fn_v(xi) for xi in x])
result=[]


#iterating over time
for n in range(nt):
    un=u.copy()

    u[1:-1] = un[1:-1] +\
    ((1j/2 * dt / dx**2) * (un[2:] - 2 * un[1:-1] + un[0:-2])) -\
    (1j*dt*v[1:-1]*un[1:-1])
    u[0]=u[-1]=0

    #normalising constant
    normal = numpy.sum(numpy.absolute(u**2))*dx
    u = u/normal

    result.append(numpy.absolute(u))

result=numpy.array(result)  #array with wavefunction u at time t

#################################################
#################################################
#################################################
#Animating the result array

fig = pyplot.figure()
axis = pyplot.axes(xlim=(-1,1), ylim = (0,10))

line, = axis.plot([],[],lw =2)

def animate(i):
    line.set_data(x,result[i+5000])
    return line,

#print(numpy.shape(result[1]),numpy.shape(x))

anim = FuncAnimation(fig, animate, frames = 300, interval = 1, blit = True)

anim.save('/home/nanarbar/Desktop/shiggydiggy.mp4',
		writer = 'ffmpeg', fps = 60)


pyplot.show()
