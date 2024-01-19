from firedrake import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


# Save the output in a PDF
def save_multi_image(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


# Read the solution
with CheckpointFile("solution-9.h5", 'r') as afile:
    mesh = afile.load_mesh("firedrake_default")
    z = afile.load_function(mesh, "U")

x, y = SpatialCoordinate(mesh)
mean = 290
st_dev = 10
sample = 100
q_range = np.random.normal(mean, st_dev, sample)
indx = np.linspace(0, pi/2, 46)
XR = np.zeros(46)
alpha = 0
for q in q_range:
    for j in range(46):
        A = assemble(inner(z.sub(0), cos(q*(x*cos(alpha) + y*sin(alpha))))*dx(degree=20))
        B = assemble(inner(z.sub(0), sin(q*(x*cos(alpha) + y*sin(alpha))))*dx(degree=20))
        C = 1/sqrt(2*pi*st_dev**2)*exp(-(q-mean)**2/(2*st_dev**2))*sqrt(A**2 + B**2)
        XR[j] += C
        alpha += pi/90
    alpha = 0

FFF = "XR.txt"
with open(FFF, "w+") as f:
    f.write("XR = %s" % (XR))

filename = "XR.pdf"
fig_1 = plt.figure()
plt.plot(indx, XR, 'o', markersize=4)
plt.xlabel('alpha in in degrees')
save_multi_image(filename)




