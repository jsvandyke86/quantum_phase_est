import numpy as np
import json
import matplotlib.pyplot as plt
import os

plt.rc('text', usetex=True)
labelsize = 16
ticksize=14
annotatesize=18
filename = "kitaevpea_xxzmodel_singletoverlaps.dat"
outfilename = 'plot_kitaev_pea_singletoverlaps_numrounds1.pdf'
infi = open(filename,'r')
filedata = infi.readlines()
infi.close()
theparams = json.loads(filedata[0][2:])
print(theparams)

initial_overlap, final_overlap, meas_outcome = np.loadtxt(filename,dtype={'formats':(np.float_,np.float_,np.int_),'names':('in','out','meas')},comments='#',unpack=True)
print(initial_overlap)
print(final_overlap)
print(meas_outcome)

fig, ax = plt.subplots()

data_curves = []
for ct in range(len(initial_overlap)):
    data_curves.append(np.array([initial_overlap[ct],final_overlap[ct]]))

for ct in range(len(initial_overlap)):
    ax.plot([0,1],data_curves[ct],'-o')
    plt.annotate(meas_outcome[ct],xy=(1,data_curves[ct][1]),xytext=(5.0,0.2),textcoords='offset points',fontsize=annotatesize)

ax.set_xticks([0,1])
ax.tick_params(axis='both', which='major', labelsize=ticksize)
ax.set_xlabel('PEA step',fontsize=labelsize)
# ax.set_ylabel(r'$|\langle S | \psi \rangle |^2$',fontsize=labelsize)
ax.set_ylabel('Overlap squared with singlet',fontsize=labelsize)
ax.set_xlim((0.0,1.1))
ax.set_ylim((0.95*np.amin(final_overlap),1.0))

mymeta = {}
mymeta["datafile"] = filename
mymeta['plotscript'] = os.path.basename(__file__)
mymeta['myfilename'] = outfilename


plt.savefig(outfilename)
plt.tight_layout()
plt.show()
