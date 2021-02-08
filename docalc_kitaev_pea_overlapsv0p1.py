# kitaev phase estimation -- look at eigenstate overlaps after process is done
# versions
# 0.1 built from docalc_kitaev_peav0p13.py

from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import PauliOp
from qiskit import QuantumCircuit, execute, Aer, QuantumRegister, ClassicalRegister

import numpy as np
import os
import json

from xxz_modelv0p2 import xxz_model

def biasedhadamard(acircuit,i,p,q):
    acircuit.u(2*np.arccos(np.sqrt(1-p/q)),0,np.pi,i)

def get_cU(hamiltonian,tval):

    # evolution operator and corresponding circuit
    U_evo = (-tval*hamiltonian).exp_i() # minus sign since exp_i implements exp{-i*op}
    U_mat = U_evo.to_matrix()



    U_evo_circ = U_evo.to_circuit()
    decomp_circ = U_evo_circ.decompose().decompose()

    backend = Aer.get_backend("unitary_simulator")
    unitary = execute(U_evo_circ, backend=backend).result().get_unitary()

    entry00 = unitary[0,0]


    backend = Aer.get_backend("unitary_simulator")
    unitary = execute(decomp_circ, backend=backend).result().get_unitary()

    entry00_2 = unitary[0,0]

    # print(unitary.shape)
    # for row in unitary:
    #     for item in row:
    #         print("{0:20.2f}".format(np.round(item,2)),end='')
    #     print(' ')

    cU_evo_circ = U_evo_circ.control()
    cU_evo_circ_fix_phase = QuantumCircuit(cU_evo_circ.num_qubits)
    cU_evo_circ_fix_phase.p(np.angle(entry00/entry00_2),0)
    cU_evo_circ_fix_phase.compose(cU_evo_circ,range(cU_evo_circ.num_qubits),inplace=True)

    return cU_evo_circ_fix_phase

def main():
    np.set_printoptions(linewidth=200)

    inputparams = {'L':2,
                    'Jxy':1.0,
                    'Jz':0.8,
                    'nbits':1,
                    'ntrials':10,
                    'operators':'spin',
                    'biasedhadamard_p':1,
                    'biasedhadamard_q':8,
                    'scriptname':os.path.basename(__file__)}

    outfi = open("kitaevpea_xxzmodel_singletoverlaps.dat","w",1)
    outfi.write("# ")
    outfi.write(json.dumps(inputparams))
    outfi.write("\n# initial state overlap^2 with singlet, final state overlap^2 with singlet, measurement outcomes\n")


    # Hamiltonian
    myHam = xxz_model(L=inputparams['L'],Jxy=inputparams['Jxy'],Jz=inputparams['Jz'],operators=inputparams['operators'])

    ham_mat = myHam.to_matrix()
    ham_evals,ham_evecs = np.linalg.eigh(ham_mat)
    print(ham_evals)
    for ct,ev in enumerate(ham_evals):
        print(ev,ham_evecs[:,ct])
        if np.absolute(ham_evecs[0,ct] - 1.0) < 1.0e-10:
            allupstate_energy = ev

    cU_circ = get_cU(myHam,np.pi)

    # XXZ hamiltonian - measure all bits, with phase corrections
    m = inputparams['nbits']

    qreg = QuantumRegister(cU_circ.num_qubits)
    cregs = [ClassicalRegister(1) for ct in range(m)]

    for trial in range(inputparams['ntrials']):
        my_circ = QuantumCircuit(qreg,*cregs)

        # create superposition of |up up> and |down down> with unequal angles -- but they are degenerate, so phase est does not change the state
        # my_circ.ry(np.pi/4,qreg[1])
        # my_circ.cx(qreg[1],qreg[2])

        # create almost singlet state
        # my_circ.h(1)
        biasedhadamard(my_circ,1,inputparams['biasedhadamard_p'],inputparams['biasedhadamard_q'])
        my_circ.cx(1,2)
        my_circ.x(1)
        my_circ.z(2)

        # calculate input state
        # print(my_circ)
        backend = Aer.get_backend("statevector_simulator")
        statevector = execute(my_circ, backend,shots=1).result().get_statevector()
        print('input state')
        for val in statevector:
            print(np.round(val,4),end=' ')
        print(" ")

        # print("overlap^2 with |up up>")
        # overlap = np.vdot(np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]),statevector)
        print("overlap^2 with |singlet>")
        overlapinitial = np.vdot(np.array([0.0,0.0,1.0/np.sqrt(2.0),0.0,-1.0/np.sqrt(2.0),0.0,0.0,0.0]),statevector)

        print(overlapinitial.conjugate()*overlapinitial)

        for t in range(m-1,-1,-1):
            my_circ.h(qreg[0])
            for ct in range(2**t):
                my_circ.compose(cU_circ,qreg,inplace=True)
            for bit in range((m-1)-t):
                my_circ.p(-np.pi/(2**(m-1-t-bit)),qreg[0]).c_if(cregs[bit],1)
            my_circ.h(qreg[0])

            my_circ.measure(qreg[0],cregs[(m-1)-t])
            my_circ.reset(qreg[0])

        # print(my_circ)
        backend = Aer.get_backend("statevector_simulator")
        results = execute(my_circ, backend,shots=1).result()
        statevector = results.get_statevector()
        counts = results.get_counts()
        print("output state")
        for val in statevector:
            print(np.round(val,4),end=' ')
        print(" ")

        # print("overlap^2 with |up up>")
        # overlap = np.vdot(np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]),statevector)
        print("overlap^2 with |singlet>")
        overlapfinal = np.vdot(np.array([0.0,0.0,1.0/np.sqrt(2.0),0.0,-1.0/np.sqrt(2.0),0.0,0.0,0.0]),statevector)


        print(overlapfinal.conjugate()*overlapfinal)
        print('counts',counts)

        for key,item in counts.items():
            pass
        outfi.write("{0:22.12e} {1:22.12e} {2}\n".format((overlapinitial.conjugate()*overlapinitial).real,(overlapfinal.conjugate()*overlapfinal).real,key.replace(" ","")))

    outfi.close()

if __name__ == '__main__':
    main()
