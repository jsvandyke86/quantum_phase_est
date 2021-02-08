# hamiltonians for the XXZ and related models in qiskit
# versions
# 0.1 using the WeightedPauliOperator class
# 0.2 switch to the new qiskit classes for operators

from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import PauliOp, SummedOp, ListOp

def nearestneighbor_interactions(L,interaction_type,interaction_strength):
    ham_terms = []
    for ct in range(L-1):
        paulilist = ['I',]*L
        paulilist[ct] = interaction_type[0]
        paulilist[ct+1] = interaction_type[1]
        paulistr = Pauli.from_label(''.join(paulilist))
        ham_terms.append(PauliOp(paulistr,interaction_strength))
    # PBC term
    if L > 2:
        paulilist = ['I',]*L
        paulilist[0] = interaction_type[0]
        paulilist[L-1] = interaction_type[1]
        paulistr = Pauli.from_label(''.join(paulilist))
        ham_terms.append(PauliOp(paulistr,interaction_strength))
    return SummedOp(ham_terms)

def xxz_model(L,Jxy,Jz,operators):
    if operators == 'pauli':
        return nearestneighbor_interactions(L,'XX',Jxy) \
                + nearestneighbor_interactions(L,'YY',Jxy) \
                + nearestneighbor_interactions(L,'ZZ',Jz)
    elif operators == 'spin':
        return (1/4)*(nearestneighbor_interactions(L,'XX',Jxy) \
                + nearestneighbor_interactions(L,'YY',Jxy) \
                + nearestneighbor_interactions(L,'ZZ',Jz))
