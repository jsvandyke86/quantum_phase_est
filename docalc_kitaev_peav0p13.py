# kitaev phase estimation
# versions
# 0.1 test on the xxz hamiltonian for L=2 (exploring the qiskit bug about global phases and how to fix it)
# 0.11 try to generalize the qiskit bug fix hack to work for L=3
# 0.12 clean up code
# 0.13 save results

from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import PauliOp
from qiskit import QuantumCircuit, execute, Aer, QuantumRegister, ClassicalRegister

import numpy as np
import os
import json

from xxz_modelv0p2 import xxz_model


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

    inputparams = {'L':3,
                    'Jxy':1.0,
                    'Jz':0.8,
                    'nbits':10,
                    'operators':'spin',
                    'scriptname':os.path.basename(__file__)}

    outfi = open("kitaevpea_xxzmodel_L"+str(inputparams['L'])+"_Jz"+str(inputparams['Jz'])+".dat","w",1)
    outfi.write("# ")
    outfi.write(json.dumps(inputparams))
    outfi.write("\n# exact energy all |up> state, estimated energy from PEA\n")


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
    my_circ = QuantumCircuit(qreg,*cregs)

    for t in range(m-1,-1,-1):
        my_circ.h(qreg[0])
        for ct in range(2**t):
            my_circ.compose(cU_circ,qreg,inplace=True)
        for bit in range((m-1)-t):
            my_circ.p(-np.pi/(2**(m-1-t-bit)),qreg[0]).c_if(cregs[bit],1)
        my_circ.h(qreg[0])

        # testing ordering of counts
        # if t==0:
        #     my_circ.x(qreg[0])
        my_circ.measure(qreg[0],cregs[(m-1)-t])
        my_circ.reset(qreg[0])

    # print(my_circ)
    backend = Aer.get_backend("qasm_simulator")
    counts = execute(my_circ, backend,shots=100).result().get_counts()
    print(counts)
    maxct = 0
    maxctkey = ' '
    for key,item in counts.items():
        if item > maxct:
            maxctkey = key
            maxct = item
    maxctkey = maxctkey.replace(" ","")
    print(maxctkey)
    est_val = 0.0
    for ct,bit in enumerate(maxctkey,start=1):
        if bit=='1':
            est_val += 1/(2**int(ct))
    print('est phase',est_val)
    est_energy = 2*est_val
    print('est energy',est_energy)

    outfi.write("{0:22.12e} {1:22.12e}\n".format(allupstate_energy,est_energy))

if __name__ == '__main__':
    main()
