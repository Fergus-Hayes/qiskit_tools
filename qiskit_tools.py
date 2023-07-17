from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer, IBMQ
from scipy.interpolate import approximate_taylor_polynomial
from qiskit.circuit.library import RGQFTMultiplier, DraperQFTAdder
from qiskit.circuit.library.basis_change import QFT as QFT_pre
from qiskit.circuit.library.standard_gates import PhaseGate, RYGate, CSwapGate
from qiskit.circuit.library.arithmetic.integer_comparator import IntegerComparator
from qiskit.circuit.library.standard_gates import XGate
from qiskit.quantum_info.operators.pauli import Pauli
from qiskit.circuit.library.boolean_logic import OR
from qiskit.quantum_info import random_unitary
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller
import itertools as it
import numpy as np
import sympy

def signmag_bin_to_dec(binary, nint=None):
    mag = bin_to_dec(binary[1:], nint=nint)
    sign = (-1)**(int(binary[0]))
    return mag*sign

def bin_to_dec(binary, nint=None, phase=False, signmag=False):
    """
    Convert a binary string to a floating point number
    binary - input binary (string)
    nint - number of integer bits. Default to all (int)
    """
    if signmag:
       return signmag_bin_to_dec(binary, nint)

    basis = 0. 

    n = len(binary)

    if phase:
        if binary[0]=='1':
            if nint is None:
                nint_ = n-1
            else:
                nint_ = nint
            basis = -(2.**(nint_))
        binary = binary[1:]

    n = len(binary)
    if nint is None:
        nint = n

    digit = 0.
    for i,bit in enumerate(np.arange(nint-n,nint)[::-1]):
        digit+=(2.**bit)*int(binary[i])
    return digit + basis

def my_binary_repr(digit, n, nint=None, phase=False, nround=True, overflow_error=True, signmag=False):
    """
    Convert a floating point digit to binary string
    digit - input number (float)
    n - total number of bits (int)
    nint - number of integer bits. Default to lowest required (int)
    """

    if signmag and phase:
        bin_out = my_binary_repr(np.abs(digit), n-1, nint=nint, phase=False, signmag=False)
        if digit<0.:
            return '1'+ bin_out
        else:
            return '0'+ bin_out

    if nint is None:# or nint==n:
        if phase:
            nint = n - 1
        else:
            nint = n

    if phase:
        p = n - nint - 1
        dmax = 2.**(nint) - 2.**(-p)
        dmin = -2.**(nint)
    else:
        p = n - nint
        dmax = 2.**(nint) - 2.**(-p)
        dmin = 0.

    if overflow_error:
        if digit>dmax or digit<dmin:
            raise ValueError('Digit '+str(digit)+' does not lie in the range:',dmin,'-',dmax,n,nint,p)

    if nround:
        n += 1
        p += 1

    value = digit
    bin_out = ''
    if phase:
        if value<0.:
            value+=2.**nint
            bin_out+='1'
        else:
            bin_out+='0'
    
    for i,bit in enumerate(np.arange(-p,nint)[::-1]):
        bin_out+=str(int(np.floor(value/2.**bit)))
        if value>=2.**bit:
            value-=2.**bit

    if nround:
        carry = True
        bin_out = np.array(list(bin_out))
        for i in np.arange(n)[::-1]:
            if not carry:
                break
            if bin_out[i]=='1':
                bin_out[i]='0'
            elif bin_out[i]=='0':
                bin_out[i]='1'
                carry = False
        bin_out = ("").join(list(bin_out[:-1]))

    return bin_out

def pres_est(digit, n, nint=None, phase=False):
    if phase:
        n -= 1

    if nint is None:
        nint = n

    if digit==0.:
        return 0.

    if digit!=0.:
        return (digit/np.abs(digit))*bin_to_dec(my_binary_repr(digit, n, nint=nint, phase=phase), nint=nint, phase=phase)
    else:
        return 0.

def twos_compliment(binary):
    n = len(binary)

    if np.sum(np.array(list(binary)).astype(int))==0:
        compliment = binary
    else:
        compliment = my_binary_repr(bin_to_dec(''.join(list(np.logical_not(np.array(list(binary)).astype(bool)).astype(int).astype(str))), nint=None, phase=False) + 1, n, nint=None, phase=False, overflow_error=False)

    return compliment

def get_nint(digits):
    if np.array(digits).ndim==0:
        digits=np.array([digits])
    digits = np.where(np.abs(digits)>1.,np.modf(digits)[1],digits)
    digits = digits[digits!=0.]
    if len(digits)==0:
        return 0
    nint = int(np.ceil(np.log2(np.max(np.abs(digits)+1))))
    return nint

def get_npres(digits):
    if np.array(digits).ndim==0:
        digits=np.array([digits])
    digdecs = np.modf(digits)[0]
    digdecs = digdecs[digdecs!=0]
    if len(digdecs)==0:
        return 0
    mindec = np.min(np.abs(digdecs))
    switch = True
    p = 0
    while switch:
        if mindec%(2.**-p)==0.:
            switch=False
        p+=1
    return p-1

def piecewise_poly(xs, coeffs, bounds_):
    ys = np.array([])
    bounds__ = np.copy(bounds_)
    bounds__[-1] = np.inf
    for i in np.arange(len(bounds__))[:-1]:
        ys_ = np.array(np.polyval(coeffs[i], xs[np.greater_equal(xs,bounds__[i])&np.greater(bounds__[i+1],xs)]))
        ys = np.concatenate([ys, ys_])
    return ys

def get_coefficients(xs, coeffs, bounds_):
    ys = np.array([])
    bounds__ = np.copy(bounds_)
    bounds__[-1] = np.inf
    for i in np.arange(len(bounds__))[:-1]:
        #print(np.where(np.greater_equal(xs,bounds[i])&np.greater(bounds[i+1],xs), coeffs[i]))
        ys_ = np.ones(np.sum(np.greater_equal(xs,bounds__[i])&np.greater(bounds__[i+1],xs)))*coeffs[i]
        #ys_ = np.array(np.polyval(coeffs[i], xs[np.greater_equal(xs,bounds[i])&np.greater(bounds[i+1],xs)]))
        ys = np.concatenate([ys, ys_])
    return ys


def get_bound_coeffs(func, bounds, norder, reterr=False):
    if np.array(bounds).ndim==0:
        print('Bounds must be a list of two entries!')
    if len(bounds)==1:
        print('Bounds must be a list of two entries!')

    coeffs = []
    errs = []
    for i in np.arange(len(bounds))[:-1]:
        coeffs_, err_ = run_remez(func, bounds[i], bounds[i+1], norder)
        coeffs.append(np.array(coeffs_))
        errs.append(err_)
    if reterr:
        return np.array(coeffs), np.array(errs)
    else:
        return np.array(coeffs)

def input_bits_to_qubits(binary, circ, reg, wrap=False, inverse=False, phase=False, qphase=None, label='Input'):
    # Flips the qubits to match a classical bit string
    
    n = len(binary)
    
    if inverse:
        wrap = True

    if wrap:
        regs = []
        reg = QuantumRegister(n, 'reg')
        regs.append(reg)
        if qphase!=None:
            qphase = QuantumRegister(1, 'phase')
            regs.append(qphase)
        circ = QuantumCircuit(*regs)

    if phase and qphase==None:
        qphase = QuantumRegister(1, 'phase')
        circ.add(qphase)

    for bit in np.arange(n):
        if int(binary[::-1][bit])==1:
            circ.x(reg[bit])

    if phase<0.:
        circ.x(qphase[0])

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'
    
    return circ

def binary_input(data_ins, n, nint=None):
    #Return the binary representation of the input number as a string
    data_outs=[]
    for data_in in data_ins:
        data_outs.append(my_binary_repr(data_in, n, nint=nint))
    return np.array(data_outs)

def qft_rotations(circuit, n):
    """Performs qft on the first n qubits in circuit (without swaps)"""
    circuit.h(n)
    for qubit in np.arange(n):
        circuit.cp(np.pi/2.**(n-qubit), qubit, n)
    # At the end of our function, we call the same function again on
    # the next qubits (we reduced n by one earlier in the function)
    return circuit

def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def qft(circuit, n):
    """QFT on the first n qubits in circuit"""
    for qubit in np.arange(n)[::-1]:
        circuit = qft_rotations(circuit, qubit)
    swap_registers(circuit, n)
    return circuit

def QFT(circ, qreg, do_swaps=True, approximation_degree=0., insert_barriers=False, wrap=False, inverse=False, label='QFT'):

    n = len(qreg)

    if inverse:
        wrap = True

    if wrap:
        qreg = QuantumRegister(n, 'q_reg')
        circ = QuantumCircuit(qreg)

    for j in reversed(np.arange(n)):
        circ.h(j)
        num_entanglements = max(0, j - max(0, approximation_degree - (n - j - 1)))
        for k in reversed(np.arange(j - num_entanglements, j)):
            lam = np.pi * (2.0 ** (k - j))
            circ.cp(lam, j, k)

        if insert_barriers:
            circ.barrier()

    if do_swaps:
        for i in np.arange(n // 2):
            circ.swap(i, n - i - 1)

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'

    return circ

def QFTAddition(circ, qreg1, qreg2, wrap=False, inverse=False, label='Add', QFT_on=True, iQFT_on=True, nint1=None, nint2=None, phase=False, signmag=False):
    r"""
    |qreg1>|qreg2> -> |qreg1>|qreg2 + qreg1>
    """
    n1 = len(qreg1)
    n2 = len(qreg2)

    #if n1>n2:
    #    raise ValueError('First register cannot be greater than the second!',n1,n2)

    if inverse:
        wrap = True

    if wrap:
        qreg1 = QuantumRegister(n1, 'q_reg1')
        qreg2 = QuantumRegister(n2, 'q_reg2')
        circ = QuantumCircuit(qreg1, qreg2)

    if nint1==None:
        nint1 = n1
    if nint2==None:
        nint2 = n2

    dp = (n2 - nint2) - (n1 - nint1)
    dpn = 0
    if dp<0:
        dpn = np.abs(dp)
        dp = 0

    add_gate = QFTAddition_(circ, qreg1[dpn:], qreg2[dp:], wrap=True, phase=False, QFT_on=QFT_on, iQFT_on=iQFT_on, signmag=signmag)
    circ.append(add_gate, [*qreg1[dpn:], *qreg2[dp:]]);

    if n1!=n2 and not signmag and phase:
        for i in np.arange(nint2-nint1):
            circ.cx(qreg1[-1], qreg2[-i-1]);

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'

    return circ

def QFTAddition_(circ, qreg1, qreg2, wrap=False, inverse=False, label='Add', QFT_on=True, iQFT_on=True, pm=1, phase=False, signmag=False):
    r"""
    |qreg1>|qreg2> -> |qreg1>|qreg2 + qreg1>
    """
    n1 = len(qreg1)
    n2 = len(qreg2)

    if inverse:
        wrap = True

    if wrap:
        qreg1 = QuantumRegister(n1, 'q_reg1')
        qreg2 = QuantumRegister(n2, 'q_reg2')
        circ = QuantumCircuit(qreg1, qreg2)

    if QFT_on:
        circ.append(QFT(circ, qreg2, do_swaps=False, wrap=True), qreg2[:])

    jend = n1
    if phase and signmag:
        circ.append(TwosCompliment(circ, qreg1[:-1], wrap=True).control(1), [qreg1[-1], *qreg1[:-1]]);
        jend -= 1

    for j in np.arange(0,jend):
        for l in np.arange(0,n2):
            lam = pm*2*np.pi*2.**(j-l-1)
            if lam%(2*np.pi)==0:
                continue
            circ.cp(lam, qreg1[j], qreg2[l])
            if phase and n1!=n2 and lam%np.pi!=0:
                circ.append(PhaseGate(-2.*lam).control(2), [qreg1[-1], qreg1[j], qreg2[l]]);

    if phase and signmag:
        circ.append(TwosCompliment(circ, qreg1[:-1], wrap=True).control(1), [qreg1[-1], *qreg1[:-1]]);

    if iQFT_on:
        circ.append(QFT(circ, qreg2, do_swaps=False, wrap=True, inverse=True), qreg2[:])

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'

    return circ

def QFTSubtraction(circ, qreg1, qreg2, wrap=False, inverse=False, label='Sub', QFT_on=True, iQFT_on=True):
    r"""
    |qreg1>|qreg2> -> |qreg1>|qreg2 - qreg1>
    """
    n1 = qreg1.size
    n2 = qreg2.size

    if inverse:
        wrap = True

    if wrap:
        qreg1 = QuantumRegister(n1, 'q_reg1')
        qreg2 = QuantumRegister(n2, 'q_reg2')
        circ = QuantumCircuit(qreg1, qreg2)

    if QFT_on:
        circ.append(QFT(circ, qreg2, do_swaps=False, wrap=True), qreg2[:])

    for j in np.arange(n1):
        for k in np.arange(n2 - j):
            lam = -np.pi / (2.**(k))
            if lam%(2*np.pi)==0.:
                continue
            circ.cp(lam, qreg1[j], qreg2[j + k])

    if iQFT_on:
        circ.append(QFT(circ, qreg2, do_swaps=False, wrap=True, inverse=True), qreg2[:])

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'

    return circ

def QFTSubtractionRHS(circ, qreg1, qreg2, wrap=False, inverse=False, label='Sub', QFT_on=True, iQFT_on=True):
    r"""
    |qreg1>|qreg2> -> |qreg1>|qreg1 - qreg2>
    """
    n1 = qreg1.size
    n2 = qreg2.size

    if inverse:
        wrap = True

    if wrap:
        qreg1 = QuantumRegister(n1, 'q_reg1')
        qreg2 = QuantumRegister(n2, 'q_reg2')
        circ = QuantumCircuit(qreg1, qreg2)

    if QFT_on:
        circ.append(QFT(circ, qreg2, do_swaps=False, wrap=True, inverse=True), qreg2[:])

    for j in np.arange(n1):
        for k in np.arange(n2 - j):
            lam = np.pi / (2.**(k))
            if lam%(2*np.pi)==0.:
                continue
            circ.cp(lam, qreg1[j], qreg2[j + k])

    if iQFT_on:
        circ.append(QFT(circ, qreg2, do_swaps=False, wrap=True, inverse=True), qreg2[:])

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'

    return circ


def OnebitSub(circ, qbit1, qbit2, qans, wrap=True, inverse=False, label='Sub1'):
    if len(qans)!=2:
        print('Ancillary register must be of 2 qubits, length:', len(qans))
        return 0

    if inverse:
        wrap = True

    if wrap:
        qbit1 = QuantumRegister(1, 'q_bit1')
        qbit2 = QuantumRegister(1, 'q_bit2')
        qans = QuantumRegister(2, 'q_ans')
        circ = QuantumCircuit(qbit1, qbit2, qans)

    circ.x(qbit2)
    circ.ccx(qbit1, qbit2, qans[0])
    circ.x(qbit1)
    circ.x(qbit2)
    circ.ccx(qbit1, qbit2, qans[1])
    circ.x(qbit1)

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'

    return circ

def BitStringCompare(circ, qreg1, qreg2, qans, wrap=False, inverse=False, label='BitComp'):
    n = qreg1.size

    if n!=qreg1.size:
        print('Registers must be of equal length to compare!')
        return 0

    if len(qans)!=(3*n)-1:
        print('Ancillary register must be 3xlength of input register - 1, length = ', len(qans))
        return 0

    if inverse:
        wrap = True

    if wrap:
        qreg1 = QuantumRegister(n, 'q_reg1')
        qreg2 = QuantumRegister(n, 'q_reg2')
        qans = QuantumRegister((3*n)-1, 'q_ans')
        circ = QuantumCircuit(qreg1, qreg2, qans)

    for i in np.arange(n):
        bit_gate = OnebitSub(circ, qreg1[i], qreg2[i], qans[3*i:(3*i)+2], wrap=True)
        circ.append(bit_gate, [qreg1[i], qreg2[i], *qans[3*i:(3*i)+2]])
        if i+1<n:
            circ.x(qans[3*i])
            circ.x(qans[(3*i)+1])
            circ.ccx(*qans[3*i:(3*i)+3])
            circ.x(qans[3*i])
            circ.x(qans[(3*i)+1])
    for i in np.arange(n)[1:][::-1]:    
            circ.ccx(qans[(3*(i-1))+2], qans[3*i], qans[3*(i-1)])
            circ.ccx(qans[(3*(i-1))+2], qans[(3*i)+1], qans[(3*(i-1))+1])

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'

    return circ

def TwosCompliment(circ, qreg, wrap=False, inverse=False, QFT_on=True, iQFT_on=True, label='comp2'):
    n = len(qreg)

    if inverse:
        wrap = True

    if wrap:
        qreg = QuantumRegister(n, 'q_reg')
        circ = QuantumCircuit(qreg)

    for qubit in qreg:
        circ.x(qubit)

    inc_gate = QFTBinaryAdd(circ, qreg, ''.join(np.zeros(n-1).astype(int).astype(str))+'1', wrap=True)
    circ.append(inc_gate, qreg)

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'

    return circ

def QFTMultiply(circ, qreg1, qreg2, qreg3, A=1., wrap=False, inverse=False, nint1=None, nint2=None, nint3=None, phase=False, label='Mult', QFT_on=True, iQFT_on=True):
    r"""
    |qreg1>|qreg2>|qreg3> -> |qreg1>|qreg2>|qreg1 x qreg2>
    """
    n1 = len(qreg1)
    n2 = len(qreg2)
    n3 = len(qreg3)
    if n3!=n1+n2 and nint3 == None:
        raise ValueError('Output register should be the combined length of both input registers if no integer bit length is specified.')

    if nint1==None:
        nint1=n1
    if nint2==None:
        nint2=n2
    if nint3==None:
        nint3=n3
    
    nshift = (nint1 + nint2)-nint3

    if phase:
        nshift+=1

    if inverse:
        wrap = True

    if wrap:
        qreg1 = QuantumRegister(n1, 'q_reg1')
        qreg2 = QuantumRegister(n2, 'q_reg2')
        qreg3 = QuantumRegister(n3, 'q_reg3')
        circ = QuantumCircuit(qreg1, qreg2, qreg3)

    if QFT_on:
        circ.append(QFT(circ, qreg3, do_swaps=False, wrap=True), qreg3[:])

    for j in np.arange(1, n1 + 1):
        for i in np.arange(1, n2 + 1):
            for k in np.arange(1, n3 + 1):
                lam = A*(2 * np.pi) / (2. ** (i + j + k - n3 - nshift))
                if lam%(2*np.pi)==0.:
                    continue
                circ.append(PhaseGate(lam).control(2),[qreg1[n1 - j], qreg2[n2 - i], qreg3[k - 1]])

    if iQFT_on:
        circ.append(QFT(circ, qreg3, do_swaps=False, wrap=True, inverse=True), qreg3[:])

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'

    return circ

def QFTMultPhase(circ, qreg1, qreg2, qreg3, wrap=False, inverse=False, nint1=None, nint2=None, nint3=None, label='MultPhase', QFT_on=True, iQFT_on=True, signmag1=False, signmag2=False, signmag3=False, poszero=False):
    r"""
    |qreg1>|qreg2>|qreg3> -> |qreg1>|qreg2>|qreg1 x qreg2>
    """
    n1 = len(qreg1)
    n2 = len(qreg2)
    n3 = len(qreg3)
    if n3>n1+n2:
        raise ValueError('The output register is greater than necessary.')
    if n3!=n1+n2 and nint3 == None:
        raise ValueError('Output register should be the combined length of both input registers if no integer bit length is specified.')

    if nint1==None:
        nint1=n1
    if nint2==None:
        nint2=n2
    if nint3==None:
        nint3=n3

    if inverse:
        wrap = True

    if wrap:
        qreg1 = QuantumRegister(n1, 'q_reg1')
        qreg2 = QuantumRegister(n2, 'q_reg2')
        qreg3 = QuantumRegister(n3, 'q_reg3')
        circ = QuantumCircuit(qreg1, qreg2, qreg3)

    # Define the twos-complement operations
    tc_gate_1 = TwosCompliment(circ, qreg1[:-1], wrap=True).control(1)
    tc_gate_2 = TwosCompliment(circ, qreg2[:-1], wrap=True).control(1)
    tc_gate_3 = TwosCompliment(circ, qreg3[:-1], wrap=True).control(2)

    # Add the twos-complement operations to convert registers 1 and 2 to sign-magnitude notation
    if not signmag1:
        circ.append(tc_gate_1, [qreg1[-1], *qreg1[:-1]]);
    if not signmag2:
        circ.append(tc_gate_2, [qreg2[-1], *qreg2[:-1]]);

    # Apply simple QFT multiplication
    mult_gate = QFTMultiply(circ, qreg1[:-1], qreg2[:-1], qreg3[:-1], nint1=nint1, nint2=nint2, nint3=nint3, wrap=True)
    circ.append(mult_gate, [*qreg1[:-1], *qreg2[:-1], *qreg3[:-1]]);

    if not signmag3:
        # If register 1 is positive, but register 2 is negative, then twos-complement operator on 3
        circ.x(qreg1[-1]);
        circ.append(tc_gate_3, [qreg1[-1], qreg2[-1], *qreg3[:-1]]);
        circ.x(qreg1[-1]);
        # If register 2 is positive, but register 1 is negative, then twos-complement operator on 3
        circ.x(qreg2[-1]);
        circ.append(tc_gate_3, [qreg1[-1], qreg2[-1], *qreg3[:-1]]);
        circ.x(qreg2[-1]);

    # If 1 is negative flip phase qubit of 3, and if 2 is negative flip phase qubit of 3
    circ.cx(qreg1[-1], qreg3[-1]);
    circ.cx(qreg2[-1], qreg3[-1]);
    
    if not signmag1:
        # Restore register 1 to original twos-complement notation
        circ.append(tc_gate_1, [qreg1[-1], *qreg1[:-1]]);
    if not signmag2:    
        # Restore register 2 to original twos-complement notation
        circ.append(tc_gate_2, [qreg2[-1], *qreg2[:-1]]);

    if poszero or not signmag3:
        # Flip register 1
        for qubit in np.arange(n1):
            circ.x(qreg1[qubit]);

        # If register 1 is all zeros and phase of 2 is negative, then flip the phase of 3
        circ.mcx([*qreg1, qreg2[-1]], qreg3[-1], mode='noancilla')

        # Flip back register A
        for qubit in np.arange(n1):
            circ.x(qreg1[qubit]);
    
        # Flip register B
        for qubit in np.arange(n2):
            circ.x(qreg2[qubit]);
    
        # If register 2 is all zeros and phase of 1 is negative, then flip the phase of 3
        circ.mcx([*qreg2, qreg1[-1]], qreg3[-1], mode='noancilla')

        # Flip back register 2
        for qubit in np.arange(n2):
            circ.x(qreg2[qubit]);

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'

    return circ

def QFTMultBinPhase(circ, qreg1, qreg3, binary, wrap=False, inverse=False, nint=None, label='MultBinPhase', QFT_on=True, iQFT_on=True):
    r"""
    |qreg1>|qreg2> -> |qreg1>|qreg1 x A>
    """

    n1 = len(qreg1)
    n2 = len(binary)
    n3 = len(qreg3)
    #if n3>n1+n2:
    #    raise ValueError('The output register is greater than necessary.')
    if n3!=n1+n2 and nint == None:
        raise ValueError('Output register should be the combined length of both input registers if no integer bit length is specified.')

    if nint==None:
        nint=n1

    if inverse:
        wrap = True

    if wrap:
        qreg1 = QuantumRegister(n1, 'q_reg1')
        qreg3 = QuantumRegister(n3, 'q_reg3')
        circ = QuantumCircuit(qreg1, qreg3)

    comp_gate1_inv = TwosCompliment(circ, qreg1[:-1], wrap=True, inverse=True).control(1)
    circ.append(comp_gate1_inv, [qreg1[-1], *qreg1[:-1]]);

    if binary[0]=='1':
        twoscomp = twos_compliment(binary)
    else:
        twoscomp = binary

    mult_gate = QFTBinaryMult(circ, qreg1[:-1], qreg3[:-1], twoscomp[1:],  nint=nint, wrap=True)
    circ.append(mult_gate, [*qreg1[:-1], *qreg3[:-1]]);

    # if reg1 was neg, but binary is pos, then flip 2comp of reg3
    # then vise versa

    if binary[0]=='1':
        circ.x(qreg1[-1]);
        comp_gate = TwosCompliment(circ, qreg3[:-1], wrap=True).control(1)
        circ.append(comp_gate, [qreg1[-1], *qreg3[:-1]]);
        circ.x(qreg1[-1]);

    elif binary[0]=='0':
        comp_gate = TwosCompliment(circ, qreg3[:-1], wrap=True).control(1)
        circ.append(comp_gate, [qreg1[-1], *qreg3[:-1]]);

    # if reg1(bin) is neg, flip sign of reg3

    circ.cx(qreg1[-1], qreg3[-1]);
    if binary[0]=='1':
        circ.x(qreg3[-1]);

    # reverse the twos comp

    comp_gate1 = TwosCompliment(circ, qreg1[:-1], wrap=True).control(1)
    circ.append(comp_gate1, [qreg1[-1], *qreg1[:-1]]);

    # flip all reg1 qubits
    # and all but last binary

    if binary[0]=='1':
        for qubit in np.arange(n1):
            circ.x(qreg1[qubit]);
        circ.mcx(qreg1, qreg3[-1], mode='noancilla')
        for qubit in np.arange(n1):
            circ.x(qreg1[qubit]);

    if np.sum(np.array(list(binary)).astype(int))==0:
        circ.cx(qreg1[-1], qreg3[-1]);

    #for qubit in np.arange(n3-1):
    #    circ.x(qreg3[qubit])
    
    #circ.mcx(qreg3[:-1], qreg3[-1], mode='noancilla')
    
    #for qubit in np.arange(n3-1):
    #    circ.x(qreg3[qubit])

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'

    return circ

def QFTBinaryAdd(circ, qreg, binary, wrap=False, inverse=False, QFT_on=True, iQFT_on=True, label='AddA'):
    r"""
    |qreg> -> |qreg + A>
    """
    n1 = len(qreg)
    n2 = len(binary)

    if inverse:
        wrap = True

    if wrap:
        qreg = QuantumRegister(n1, 'q_reg1')
        circ = QuantumCircuit(qreg)

    if QFT_on:
        circ.append(QFT(circ, qreg, do_swaps=False, wrap=True), qreg[:])

    for i in np.arange(1, n1 + 1):
        for k in np.arange(1, n2 + 1):
            lam = (2. * np.pi) / (2. ** (i + k - n2))
            if lam%(2*np.pi)==0.:
                continue
            if binary[i-1]=='1':
                circ.p(lam,qreg[k-1])

    if iQFT_on:
        circ.append(QFT(circ, qreg, do_swaps=False, wrap=True, inverse=True), qreg[:])

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'

    return circ

def QFTBinarySub(circ, qreg, binary, wrap=False, inverse=False, label='SubA'):
    r"""
    |qreg> -> |qreg - A>
    """
    n1 = len(qreg)
    n2 = len(binary)

    if inverse:
        wrap = True

    if wrap:
        qreg = QuantumRegister(n1, 'q_reg1')
        circ = QuantumCircuit(qreg)

    circ.append(QFT(circ, qreg, do_swaps=False, wrap=True), qreg[:])

    for i in np.arange(1, n2 + 1):
        for k in np.arange(1, n1 + 1):
            lam = -(2. * np.pi) / (2. ** (i + k - n1))
            if lam%(2*np.pi)==0.:
                continue
            if binary[k-1]=='1':
                circ.p(lam,qreg[i-1])

    circ.append(QFT(circ, qreg, do_swaps=False, wrap=True, inverse=True), qreg[:])

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'

    return circ

def QFTBinarySubRHS(circ, qreg, binary, wrap=False, inverse=False, label='SubA'):
    r"""
    |qreg> -> |A - qreg>
    """
    n1 = qreg.size
    n2 = len(binary)

    if inverse:
        wrap = True

    if wrap:
        qreg = QuantumRegister(n1, 'q_reg1')
        circ = QuantumCircuit(qreg)

    circ.append(QFT(circ, qreg, do_swaps=False, wrap=True, inverse=False), qreg[:])

    for i in np.arange(1, n1 + 1):
        for k in np.arange(1, n2 + 1):
            lam = -(2. * np.pi) / (2. ** (i + k - n2))
            if lam%(2*np.pi)==0.:
                continue
            if binary[i-1]=='1':
                circ.p(lam,qreg[k-1])

    circ.append(QFT(circ, qreg, do_swaps=False, wrap=True, inverse=True), qreg[:])

    for i in np.arange(1, n2):
       circ.x(qreg[i])

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'

    return circ

def qubit_to_phase(circ, qreg, nint=None, wrap=False, inverse=False, phase=False, factor=1, label='Amp2Phase'):
    r"""
    |qreg> -> e^(2pi i qreg)|qreg>
    """
    n = len(qreg)

    if inverse:
        wrap = True

    if wrap:
        qreg = QuantumRegister(n, 'q_reg')
        circ = QuantumCircuit(qreg)

    if phase:
        n -= 1

    if nint==None:
        nint = n

    for k in np.arange(1,n-nint+1):
        lam = 2.*np.pi*(2.**(-k))*factor
        qubit = n-nint-k
        circ.p(lam,qreg[qubit])

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'
    
    return circ

def CRotation(circ, qreg, qtarg, nint=None, phase=False, label='CRot', wrap=False, inverse=False):

    n = len(qreg)
    
    if nint is None:
        nint = n

    if inverse:
        wrap = True

    if wrap:
        qreg = QuantumRegister(n, 'qreg')
        qtarg = QuantumRegister(1, 'q_targ')

        circ = QuantumCircuit(qreg, qtarg)
        
    for i in np.arange(n):
        circ.cry(2.**(i+1+nint-n), qreg[i], qtarg)
    
    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'

    return circ   

def QFTPowerN(circ, qreg1, qreg2, N, wrap=False, inverse=False, nintx=None, nint=None, label='PowN', QFT_on=True, iQFT_on=True):
    r"""
    |qreg1>|qreg2> -> |qreg1>|qreg1**N>
    """
    n1 = qreg1.size
    n2 = qreg2.size

    if n2>(n1*N):
        raise ValueError('The output register is greater than necessary.')
    if n2!=(n1*N) and nint == None:
        raise ValueError('Output register should be '+str(N)+'x the length of the input register if no integer bit length is specified.')

    if nintx==None:
        nintx=0
    if nint==None:
        nint=0

    nshift = N-1+(N*nintx)-nint

    if inverse:
        wrap = True

    if wrap:
        qreg1 = QuantumRegister(n1, 'q_reg1')
        qreg2 = QuantumRegister(n2, 'q_reg2')
        circ = QuantumCircuit(qreg1, qreg2)

    if QFT_on:
        circ.append(QFT(circ, qreg2, do_swaps=False, wrap=True), qreg2[:])

    qubit_grid = np.meshgrid(*[np.arange(1,n1+1) for i in np.arange(N)], indexing='ij')
    grid_shape = [n1 for i in np.arange(N)]
    grid_shape.append(n2)
    gridouts = np.ones(grid_shape)*np.arange(1,n2+1)
    gridouts = 2*np.pi/(np.power(2.,np.sum(qubit_grid,axis=0)+gridouts.T-n2-nshift))

    cond = gridouts%(2.*np.pi)!=0.

    inds = np.argwhere(cond)+1
    lams = gridouts[cond]

    for j,ind in enumerate(inds):
        k = ind[0]
        ind_ = ind[1:]
        cnt = len(np.unique(ind_))
        cqubits = [n1-i for i in np.unique(ind_)]
        cqubits = [qreg1[n1-i] for i in np.unique(ind_)]
        cqubits.append(qreg2[k-1])
        circ.append(PhaseGate(lams[j]).control(cnt),cqubits)

    if iQFT_on:
        circ.append(QFT(circ, qreg2, do_swaps=False, wrap=True, inverse=True), qreg2[:])

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'†'

    return circ

def QFTDivision(circ, qreg1, qreg2, c_div, acc, backend=Aer.get_backend('qasm_simulator'), shots=10, wrap=False, inverse=False, label='DivInt'):
    if inverse:
        wrap = True

    if wrap:
        qreg1 = QuantumRegister(qreg1.size, 'q_reg1')
        qreg2 = QuantumRegister(qreg2.size, 'q_reg2')
        acc = QuantumRegister(acc.size, 'acc')
        c_div = ClassicalRegister(c_div.size, 'c_div')
        circ = QuantumCircuit(qreg1, qreg2, acc, c_div)
    
    d = QuantumRegister(1, 'd')
    circ.add_register(d)
    circ.x(d[0])

    c_dividend_str = '0'

    while c_dividend_str[0] == '0':
        print(c_dividend_str)
        circ = QFTSubtraction(circ, qreg1, qreg2)
        circ = QFTAddition(circ, acc, d)

        circ.measure(qreg1[:], c_div[:])
        
        result = execute(circ, backend=backend, shots=shots).result()

        counts = result.get_counts()
        c_dividend_str = list(counts.keys())[0]
        print(c_dividend_str)

    circ = QFTSubtract(circ, acc, d)
    
    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'
    
    return circ

def taylor_coeffs(f_, args=[], a=0., norders=3):

    x = sympy.Symbol('x')
    f = f_(x)

    coeffs = []
    for norder in np.arange(norders):
        coeffs.append(float(sympy.diff(f, x, norder).subs(x, a) / sympy.factorial(norder)))

    return np.array(coeffs)

def Gaussian_noise_amp(circ, qreg, reps=2, wrap=False, inverse=False, label='GaussNoise'):
    
    n = qreg.size

    if inverse:
        wrap = True

    if wrap:
        qreg = QuantumRegister(n, 'q_reg')
        circ = QuantumCircuit(qreg)

    for level in np.arange(reps+1):
        for i in np.arange(n-1):
            for j in np.arange(i+1,n):
                rand_op = random_unitary(4)
                circ.append(rand_op,[qreg[i],qreg[j]])

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'
    
    return circ

def depolarisation_channel(circ, qreg, p, wrap=False, inverse=False, label='depol_channel'):
    
    n = qreg.size
    
    if inverse:
        wrap = True

    if wrap:
        qreg = QuantumRegister(n, 'q_reg')
        circ = QuantumCircuit(qreg)

    num_terms = 4**n
    max_param = num_terms / (num_terms - 1)

    if p < 0 or p > max_param:
        raise NoiseError("Depolarizing parameter must be in between 0 "
                         "and {}.".format(max_param))    
    
    prob_iden = 1 - p / max_param
    prob_pauli = p / num_terms
    probs = [prob_iden] + (num_terms - 1) * [prob_pauli]
    
    paulis = [Pauli("".join(tup)) for tup in it.product(['I', 'X', 'Y', 'Z'], repeat=n)]
    
    gates_ind = np.random.choice(num_terms, p=probs, size=1)[0]
        
    gates = paulis[gates_ind]
    
    circ.append(gates, qreg[:])            
        
    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'
        
    return circ

def increment_gate(circ, qreg, wrap=False, inverse=False, QFT_on=True, iQFT_on=True, ncut=0, label='inc'):
    n = len(qreg)

    if inverse:
        wrap = True

    if wrap:
        qreg = QuantumRegister(n, 'q_reg')
        circ = QuantumCircuit(qreg)
    
    if n<ncut:
        for i in np.arange(n)[1:][::-1]:
            if i!=0:
                xgate = XGate().control(i)
                circ.append(xgate, [*qreg[:i+1]])
    
        circ.x(qreg[0])

    else:
        bin_one = ''.join(np.zeros(n-1).astype(int).astype(str))+'1'
        inc_gate = QFTBinaryAdd(circ, qreg, bin_one, wrap=True, QFT_on=QFT_on, iQFT_on=iQFT_on)
        circ.append(inc_gate, qreg)

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'

    return circ

def integer_compare(circ, qreg, qtarg, qans, value, geq=True, wrap=False, inverse=False, uncomp=True, label='intP'):
    
    n = len(qreg)

    if len(qans)!=n-1:
        raise ValueError('Ancilla register must be 1 qubit fewer than input register.')
    
    if len(qtarg)!=1:
        raise ValueError('Target register must be of 1 qubit.')
    
    if wrap:
        qreg = QuantumRegister(n, 'q_reg')
        qans = QuantumRegister(n, 'q_ans')
        circ = QuantumCircuit(qreg, qans)
        qtarg = qans[0]
        qans = qans[1:]
    
    if value<=0.:
        if geq:
            circ.x(qtarg);
    elif value < np.power(2,n):
        if n>1:
            twos = np.array(list(twos_compliment(my_binary_repr(value, n=n, phase=False))))[::-1]
            for i in np.arange(n):
                if i==0:
                    if twos[i]=='1':
                        circ.cx(qreg[i], qans[i])
                elif i<n-1:
                    if twos[i]=='1':
                        circ.compose(OR(2), [qreg[i], qans[i-1], qans[i]], inplace=True);
                    else:
                        circ.ccx(qreg[i], qans[i - 1], qans[i]);
                else:
                    if twos[i]=='1':
                        circ.compose(OR(2), [qreg[i], qans[i-1], qtarg], inplace=True);
                    else:
                        circ.ccx(qreg[i], qans[i - 1], qtarg);
            if not geq:
                circ.x(qtarg);
            
            if uncomp:
                for i in np.arange(n-1)[::-1]:
                    if i==0:
                        if twos[i]=='1':
                            circ.cx(qreg[i], qans[i]);
                    else:
                        if twos[i]=='1':
                            circ.compose(OR(2), [qreg[i], qans[i-1], qans[i]], inplace=True);
                        else:
                            circ.ccx(qreg[i], qans[i-1], qans[i]);
        else:
            circ.cx(qreg[0], qtarg);
        
            if not geq:
                circ.x(qtarg);
    else:
        if not geq:
            circ.x(qtarg);

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'

    return circ

def inequal_cond(circ, qreg, qtarg, qans, value, nint=None, phase=False, comp2=True, geq=True, wrap=False, inverse=False, uncomp=True, xflip=True, label='P'):
    
    n = len(qreg)

    if len(qans)!=n-1:
        raise ValueError('Ancilla register must be 1 qubit fewer than input register.')

    if len(qtarg)!=1:
        raise ValueError('Target register must be of 1 qubit.')

    if phase and not comp2:
        raise ValueError('Only twos-compliment representation for signed numbers is currently implemented')

    if wrap:
        qreg = QuantumRegister(n, 'q_reg')
        qans = QuantumRegister(n-1, 'q_ans')
        qtarg = QuantumRegister(1, 'q_targ')
        circ = QuantumCircuit(qreg, qtarg, qans)

    binary = my_binary_repr(value, n, phase=phase, nint=nint)
   
    if phase:
        if xflip:
            circ.x(qreg[-1]);
        if binary[0]=='0':
            binary = '1'+binary[1:]
        elif binary[0]=='1':
            binary = '0'+binary[1:]

    int_value = bin_to_dec(binary, nint=None, phase=False)

    intcomp_gate = integer_compare(circ, qreg, qtarg, qans, int_value, geq=geq, wrap=wrap, inverse=inverse, uncomp=uncomp)
    circ.append(intcomp_gate, [*qreg, *qtarg, *qans]);

    if phase and xflip:
        circ.x(qreg[-1]);

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'

    return circ

def label_gate(circ, qreg, qtarg, qans, qlab, bounds=None, wrap=False, nint=None, inverse=False, phase=False, ncut=0, label='LABEL'):
    n = len(qreg)

    if nint is None:
        nint = n
        if phase:
            nint -= 1

    if inverse:
        wrap = True

    if bounds is None:
        bounds = [0, 2**n]

    nlab = int(np.ceil(np.log2(len(bounds))))

    if nlab!=qlab.size:
        raise ValueError('Size of label register does not match the number of bounds placed.')

    if len(qans)!= n-1:
        raise ValueError('Ancilla register must have one fewer qubit than input register.')

    if wrap:
        qreg = QuantumRegister(n, 'q_reg')
        qtarg = QuantumRegister(1, 'q_targ')
        qans = QuantumRegister(n-1, 'q_ans')
        qlab = QuantumRegister(nlab, 'q_lab')
        circ = QuantumCircuit(qreg, qtarg, qans, qlab)

    circ.x(qreg[-1]);

    if nlab>=ncut:
        circ.append(QFT(circ, qlab, do_swaps=False, wrap=True), qlab)

    for i,bound_ in enumerate(bounds):

        binary = my_binary_repr(bound_, n=n, nint=nint, phase=phase)
        if binary[0]=='0':
            binary = '1'+binary[1:]
        elif binary[0]=='1':
            binary = '0'+binary[1:]

        bound = bin_to_dec(binary, nint=None, phase=False)

        intcomp_gate = integer_compare(circ, qreg, qtarg, qans, bound, geq=True, wrap=True, uncomp=False, label='P'+str(i))
        circ.append(intcomp_gate, [*qreg, qtarg[0], *qans[:]])

        inc_gate = increment_gate(circ, qlab, wrap=True, label='SET'+str(i), ncut=ncut, QFT_on=False, iQFT_on=False).control(1)
        circ.append(inc_gate, [qtarg[0], *qlab[:]])

        intcomp_gate_inv = integer_compare(circ, qreg, qtarg, qans, bound, geq=True, wrap=True, uncomp=False, inverse=True, label='P'+str(i))
        circ.append(intcomp_gate_inv, [*qreg, qtarg[0], *qans[:]])

    if nlab>=ncut:
        circ.append(QFT(circ, qlab, do_swaps=False, wrap=True, inverse=True), qlab)

    circ.x(qreg[-1]);

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'

    return circ

def cin_gate(circ, qcoff, qlab, coeffs_in, nint=None, phase=False, wrap=False, inverse=False, label='X', comp2=True):
    
    n = len(qcoff)
    nlab = len(qlab)
    
    if 2**nlab<len(coeffs_in):
        print('Not enough label states to coefficents.')
        return 0

    if inverse:
        wrap = True

    if wrap:
        qcoff = QuantumRegister(n, 'q_reg')
        qlab = QuantumRegister(nlab, 'q_lab')
        circ = QuantumCircuit(qcoff, qlab)  
    
    for i in np.arange(len(coeffs_in)):
        control_bits = my_binary_repr(i, nlab, nint=nlab, phase=False)
        if i>0:
            control_bits_ = my_binary_repr(i-1, nlab, nint=nlab, phase=False)[::-1]
        else:
            control_bits_ = np.ones(nlab).astype(int).astype(str)
        
        for j,control_bit in enumerate(control_bits[::-1]):
            if control_bit=='0' and control_bits_[j]=='1':
                circ.x(qlab[j])
        
        if comp2:
            input_gate = input_bits_to_qubits(my_binary_repr(coeffs_in[i], n, nint=nint, phase=phase), circ, reg=qcoff, wrap=True).control(nlab)
            circ.append(input_gate, [*qlab, *qcoff]);
        else:
            input_gate = input_bits_to_qubits(my_binary_repr(np.abs(coeffs_in[i]), n-1, nint=nint, phase=False), circ, reg=qcoff[:-1], wrap=True).control(nlab)
            circ.append(input_gate, [*qlab, *qcoff[:-1]]);
            if coeffs_in[i]<0.:
                circ.append(XGate().control(nlab), [*qlab, qcoff[-1]]);

        if i<len(coeffs_in)-1:
            control_bits_ = my_binary_repr(i+1, nlab, nint=nlab, phase=False)[::-1]
        else:
            control_bits_ = np.ones(nlab).astype(int).astype(str)
        
        for j,control_bit in enumerate(control_bits[::-1]):
            if control_bit=='0' and control_bits_[j]=='1':
                circ.x(qlab[j])
    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'
    
    return circ

def classic_piecewise_function(x_, coeffs_, n, bounds, nint=None, phase=False):
    
    nintx = int(np.ceil(np.log2(np.max(bounds))))
    
    x = pres_est(x_, n, nint=n, phase=phase)
    
    coeff_ind = np.argwhere(bounds>=x)[0].flatten()
    if coeff_ind!=0:
        coeff_ind = coeff_ind-1
    
    coeffs_ = coeffs_.T[coeff_ind].flatten()
    
    nintc = int(np.ceil(np.log2(np.max(np.abs(coeffs_)))))
    
    coeffs = []
    for coeff in coeffs_:
        coeffs.append(pres_est(coeff, n, nint=nintc, phase=phase))
    coeffs = np.array(coeffs)[::-1]
            
    y = pres_est(x*coeffs[0], n, nint=nint, phase=phase)
    y = pres_est(y + coeffs[1], n, nint=nint, phase=phase)
    
    for i in np.arange(2,len(coeffs)):
        y = pres_est(y*x, n, nint=nint, phase=phase)
        y = pres_est(y + coeffs[i], n, nint=nint, phase=phase)
        
    return y

def piecewise_function_posmulti(circ, q_x, q_y, q_lab, q_coff, coeffs, bounds, nint=None, nintx=None, nintcs=None, phase=False, wrap=False, inverse=False, unlabel=False, unfirst=False, comp2=False, label='f_x'):

    nlab = int(np.ceil(np.log2(len(bounds))))
    nx = len(q_x)
    n = len(q_y)
    nc = len(q_coff)

    Nord, Ncoeffs = coeffs.shape

    if Nord!=2:
        raise ValueError('Currently only working for linear piecewise approximations.')

    if nint is None:
        nint = n

    if nlab!=len(q_lab):
        raise ValueError('Size of label register is smaller than number of bounds.')
        return 0

    if np.any(nintcs==None):
        nintcs = []
        for coeffs_ in coeffs:
            nintcs.append(int(np.ceil(np.log2(np.max(np.abs(coeffs_))))))
        nintcs[-1] = nint
        nintcs = np.array(nintcs).astype(int)

    if nintx is None:
        nintx = int(np.ceil(np.log2(np.max(np.abs(bounds)))))

    if inverse:
        wrap = True

    if wrap:
        q_x = QuantumRegister(nx, 'q_x')
        q_y = QuantumRegister(n, 'q_y0')
        q_lab = QuantumRegister(nlab, 'q_lab')
        q_coff = QuantumRegister(nc, 'q_coff')

        circ = QuantumCircuit(q_x, q_y, q_lab, q_coff)

    q_ans = [*q_coff, *q_y]

    if len(q_ans)<nx:
        raise ValueError('Coefficient/output register must be greater than half the length of the x register.')

    lab_gate = label_gate(circ, q_x, q_ans[0], q_ans[1:nx+2], q_lab, bounds=bounds, nint=nintx, phase=False, wrap=True)
    circ.append(lab_gate, [*q_x, *q_ans[:nx+2], *q_lab]);

    #y1in_gate = cin_gate(circ, q_y, q_lab, coeffs[1], nint=nintcs[0,1], phase=phase, wrap=True)
    #circ.append(y1in_gate, [*q_y, *q_lab]);

    #circ.append(QFT(circ, q_y, do_swaps=False, wrap=True), q_y);

    for i in np.arange(1, Nord):

        y0in_gate = cin_gate(circ, q_coff, q_lab, coeffs[i-1], nint=nintcs[i-1,0], phase=phase, wrap=True, comp2=comp2)
        circ.append(y0in_gate, [*q_coff, *q_lab]);

        mul_gate = QFTPosMultiplicand(circ, q_coff, q_x, q_y, wrap=True, nint1=nintcs[i-1,0], nint2=nintx, nint3=nint, iQFT_on=True, QFT_on=True, comp2=comp2)
        circ.append(mul_gate, [*q_coff, *q_x, *q_y]);

        if unfirst or True:
            y0in_gate_inv = cin_gate(circ, q_coff, q_lab, coeffs[i-1], nint=nintcs[i-1,0], phase=phase, wrap=True, comp2=comp2, inverse=True)
            circ.append(y0in_gate_inv, [*q_coff, *q_lab]);

        y1in_gate = cin_gate(circ, q_coff, q_lab, coeffs[i], nint=nintcs[i-1,1], phase=phase, wrap=True)
        circ.append(y1in_gate, [*q_coff, *q_lab]);

        add_gate = QFTAddition(circ, q_coff, q_y, wrap=True, phase=phase, nint1=nintcs[i-1,1], nint2=nint, QFT_on=True, iQFT_on=True)
        circ.append(add_gate, [*q_coff, *q_y]);

    #circ.append(QFT(circ, q_y, do_swaps=False, wrap=True, inverse=True), q_y);

    if unlabel:
        lab_gate_inv = label_gate(circ, q_x, q_ans[0], q_ans[1:nx+2], q_lab, bounds=bounds, nint=nintx, phase=False, wrap=True, inverse=True)
        circ.append(lab_gate_inv, [*q_x, *q_ans[:nx+2], *q_lab]);

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'

    return circ

def SWAP_test(circ, q_1, q_2, q_a, wrap=False, inverse=False, label='SWAP'):

    n = len(q_1)

    if len(q_1)!=len(q_2):
        raise ValueError('The two registers should be the same size!')

    if inverse:
        wrap = True

    if wrap:
        q_1 = QuantumRegister(n, 'q_1')
        q_2 = QuantumRegister(n, 'q_2')
        q_a = QuantumRegister(1, 'q_a')
        circ = QuantumCircuit(q_1, q_2, q_a)

    circ.h(q_a[0])

    for qubit in np.arange(n):
        circ.cswap(q_a,q_1[qubit],q_2[qubit])

    circ.h(q_a[0])

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'

    return circ

def gate_decompose(qc):
    """decompose circuit to evaluate cost"""
    pass_ = Unroller(["u3", "cx"])
    return PassManager(pass_).run(qc).count_ops()

def Grover_Rudolph_load(circ, qx, probs, wrap=False, inverse=False, nstart=0, nend=None, label='GR_load'):

    nx = len(qx)

    if nend==None:
        nend=nx

    if len(probs)!=2**nx:
        raise ValueError('Probabilities must equal length of x register')

    if inverse:
        wrap = True

    if wrap:
        qx = QuantumRegister(nx, 'q_x')
        circ = QuantumCircuit(qx)

    for m in np.arange(nstart, nend):

        def GR_func(j):
            j = np.array(j).astype(int)
            As = []
            for i in np.arange(2**m):
                As.append(np.sum(probs[i*2**(nx-m):(i+1)*2**(nx-m)]))
            As1 = []
            for i in np.arange(2**(m+1)):
                As1.append(np.sum(probs[i*2**(nx-(m+1)):(i+1)*2**(nx-(m+1))]))
            return np.arccos(np.sqrt(np.array(As1)[::2][j]/np.array(As)[j]))

        if m==0:
            coeffs = np.arccos(np.sqrt(np.sum(probs[:2**(nx-1)])))
            circ.ry(2*coeffs, qx[nx-m-1]);

        else:
            js = np.arange(2**m)
            coeffs = GR_func(js)
            for i in np.arange(2**m):
                control_bits = my_binary_repr(i, m, nint=None, phase=False)

                if i>0:
                    control_bits_ = my_binary_repr(i-1, m, nint=None, phase=False)[::-1]
                else:
                    control_bits_ = np.ones(m).astype(int).astype(str)

                for j,control_bit in enumerate(control_bits[::-1]):
                    if control_bit=='0' and control_bits_[j]=='1':
                        circ.x(qx[nx-j-1]);

                R_gate = RYGate(2*coeffs[i]).control(int(m))
                circ.append(R_gate, [*qx[nx-m-1:][::-1]]);

                if i<2**m-1:
                    control_bits_ = my_binary_repr(i+1, m, nint=None, phase=False)[::-1]
                else:
                    control_bits_ = np.ones(m).astype(int).astype(str)

                for j,control_bit in enumerate(control_bits[::-1]):
                    if control_bit=='0' and control_bits_[j]=='1':
                        circ.x(qx[nx-j-1]);

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'

    return circ

def Grover_Rudolph_func(circ, qx, qanc, qlab, qcoff, probs, wrap=False, inverse=False, mtol=1, mmax=None, norder=1, label='GR_func'):

    nx = len(qx)
    n = len(qcoff)
    nanc = len(qanc)
    nlab = len(qlab)
    
    if mmax==None:
        mmax=nx

    if n!=nanc:
        raise ValueError('I think ancilla and coefficient reg should be the same size')
    
    if len(probs)!=2**nx:
        raise ValueError('Probabilities must equal length of x register')

    if inverse:
        wrap = True

    if wrap:
        qx = QuantumRegister(nx, 'q_x')
        qanc = QuantumRegister(nanc, 'q_anc')
        qlab = QuantumRegister(nlab, 'q_lab')
        qcoff = QuantumRegister(nanc, 'q_coff')
        circ = QuantumCircuit(qx, qanc, qlab, qcoff)

    if mtol>nx:
        GRL_gate = Grover_Rudolph_load(circ, qx, probs, wrap=True)
        circ.append(GRL_gate, qx);

    elif mtol!=0:
        probs_ = []
        for i in np.arange(2**mtol):
            probs_.append(np.sum(probs[i*2**(nx-mtol):(i+1)*2**(nx-mtol)]))
        probs_ = np.array(probs_)

        GRL_gate = Grover_Rudolph_load(circ, qx[nx-mtol:], probs_, wrap=True)
        circ.append(GRL_gate, qx[nx-mtol:]);


    for m in np.arange(mtol,mmax):
        def GR_func(j):
            j = np.array(j).astype(int)
            As = []
            for i in np.arange(2**m):
                As.append(np.sum(probs[i*2**(nx-m):(i+1)*2**(nx-m)]))
            As1 = []
            for i in np.arange(2**(m+1)):
                As1.append(np.sum(probs[i*2**(nx-(m+1)):(i+1)*2**(nx-(m+1))]))
            return np.arccos(np.sqrt(np.array(As1)[::2][j]/np.array(As)[j]))

        js = np.arange(2**m)
        coeffs = GR_func(js)
        
        bounds_ = np.linspace(0,2**m,(2**nlab)+1).astype(int)
        coeffs = get_bound_coeffs(GR_func, bounds_, norder, reterr=False).T#[::-1]
        bounds = bounds_[1:]
        
        max_list0 = np.array([coeffs[0], coeffs[1], coeffs[0]*2**nx, (coeffs[0]*2**nx)+coeffs[-1]])
        max_list1 = max_list0
        nintcs = []
        nintcs.append(int(np.ceil(np.log2(np.max(np.abs(max_list0))))))
        nintcs.append(int(np.ceil(np.log2(np.max(np.abs(max_list1))))))
        nint = nintcs[-1]
        nintcs = np.array([nintcs])
        
        func_gate = piecewise_function_posmulti(circ, qx[nx-m:], qanc, qlab, qcoff, coeffs, bounds, nint=nint, nintx=nx, nintcs=nintcs, phase=False, wrap=True, unlabel=False, unfirst=False)
        circ.append(func_gate, [*qx[nx-m:], *qanc, *qlab, *qcoff]);

        rot_gate = CRotation(circ, qanc, qx[nx-m-1], nint=nint, wrap=True)
        circ.append(rot_gate, [*qanc, qx[nx-m-1]])

        func_gate_inv = piecewise_function_posmulti(circ, qx[nx-m:], qanc, qlab, qcoff, coeffs, bounds, nint=nint, nintx=nx, nintcs=nintcs, phase=False, wrap=True, inverse=True, unlabel=False, unfirst=False)
        circ.append(func_gate_inv, [*qx[nx-m:], *qanc, *qlab, *qcoff]);

    for m in np.arange(mmax,nx):
        circ.h(qx[nx-m-1]);

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'

    return circ


def PQC_realamp(circ, qreg, weights, wrap=False, inverse=False, label='PQCrealamp'):
    n = len(qreg)
    reps = weights.shape[0] - 1

    if weights.shape[1]!=n:
        raise ValueError('Shape of weight array does not match the number of qubits of the target register.')

    if inverse:
        wrap = True

    if wrap:
        qreg = QuantumRegister(n, 'q_reg')
        circ = QuantumCircuit(qreg)

    circ.h(qreg);

    for rep in np.arange(reps+1):
        for i in np.arange(n):
            circ.ry(weights[rep,i],qreg[i])
        for i in np.arange(n):
            if i!=n-1 and rep!=reps:
                circ.cx(qreg[i], qreg[i+1])

    if wrap:
        circ = circ.to_gate()
        circ.label = label

    if inverse:
        circ = circ.inverse()
        circ.label = label+'\dag'

    return circ

def round_sig(xs, sigfig=0):
    if np.array(xs).ndim==0:
        xs = np.array([xs])
    rxs = []
    for x in xs:
        if x!=0.:
            rxs.append(np.round(x, sigfig-int(np.floor(np.log10(np.abs(x))))))
        else:
            rxs.append(0.)
    rxs = np.array(rxs)
    return rxs


def optimize_coeffs_qubits_old(func, nx, nlab, nintx, ncut0, ncut1, nsig0=4, nsig1=4, norder=1, phase=True):

    xmax = np.power(2.,nintx) - np.power(2.,nintx-nx)
    xmin = 0.
    xs = np.linspace(xmin,xmax,2**(nx))

    Nbounds = 2**nlab

    ############ Set piecewise polynomial bounds #################

    bounds_ = np.linspace(xmin, xmax, Nbounds+1)

    bounds__ = []
    for bound in bounds_:
        bounds__.append(bin_to_dec(my_binary_repr(bound, n=nx, nint=nintx, phase=False), nint=nintx, phase=False))
    bounds_ = bounds__

    coeffs = get_bound_coeffs(func, bounds_, norder, reterr=False).T
    bounds = np.array(bounds_[:-1])

    # Round bounds to given significant figures
    coeffs[0] = round_sig(coeffs[0], nsig0)
    coeffs[1] = round_sig(coeffs[1], nsig1)

    nlab = int(np.ceil(np.log2(len(bounds))))

    ###################### Playground ################################

    nint1 = get_nint(coeffs[0])
    nint2 = nintx + nint1
    nint3 = get_nint(coeffs[1])

    npres1 = get_npres(coeffs[0])
    npres2 = (nx - nintx) + npres1
    npres3 = get_npres(coeffs[1])

    n1 = npres1 + nint1 + 1
    n2 = npres2 + nint2 + 1
    n3 = npres3 + nint3 + 1

    ########### round gradients #######################

    rcoeffs = []
    for coeff in coeffs[0]:
        bitstr = my_binary_repr(coeff, 100, nint=nint1, phase=True)
        if bitstr[ncut0]=='0':
            rem = 0.
        else:
            rem = 2**(-(ncut0-nint1-1))
        if bitstr[0]=='1':
            rem = rem*-1
        rcoeff1 = bin_to_dec(bitstr[:ncut0], nint=nint1, phase=True)+rem
        rcoeff2 = bin_to_dec(bitstr[:ncut0], nint=nint1, phase=True)
        rcoeff = np.array([rcoeff1,rcoeff2])[np.argmin(np.abs([rcoeff1-coeff,rcoeff2-coeff]))]
        rcoeffs.append(rcoeff)
    rcoeffs = np.array(rcoeffs)
    coeffs[0] = rcoeffs

    fdifs = func(xs) - piecewise_poly(xs, np.array([coeffs[0],np.zeros(len(coeffs[1]))]).T, bounds_)
    coeffs_ = []
    bounds__ = bounds_
    bounds__[-1] = np.inf
    for i in np.arange(len(bounds__))[:-1]:
        coeffs_.append(np.mean(fdifs[np.greater_equal(xs,bounds__[i])&np.greater(bounds__[i+1],xs)]))
    coeffs[1] = np.array(coeffs_)
    coeffs[1] = round_sig(coeffs[1], nsig1)
    nint3 = get_nint(coeffs[1])
    npres3 = get_npres(coeffs[1])
    n3 = npres3 + nint3 + 1

    rcoeffs = []
    for coeff in coeffs[1]:
        bitstr = my_binary_repr(coeff, 100, nint=nint3, phase=True)
        if bitstr[ncut1]=='0':
            rem = 0.
        else:
            rem = 2**(-(ncut1-nint3-1))
        if bitstr[0]=='1':
            rem = rem*-1
        rcoeff1 = bin_to_dec(bitstr[:ncut1], nint=nint3, phase=True)+rem
        rcoeff2 = bin_to_dec(bitstr[:ncut1], nint=nint3, phase=True)
        rcoeff = np.array([rcoeff1,rcoeff2])[np.argmin(np.abs([rcoeff1-coeff,rcoeff2-coeff]))]
        rcoeffs.append(rcoeff)
    rcoeffs = np.array(rcoeffs)

    coeffs[1] = rcoeffs

    ############## and repeat ########################

    A1x = piecewise_poly(xs, np.array([coeffs[0],np.zeros(len(coeffs[1]))]).T, bounds_)
    A1x_A0 = piecewise_poly(xs, coeffs.T, bounds_)
    coeffs_old = np.copy(coeffs)

    coeffs[0] = np.array([*coeffs[0,2**(nlab-1)+1:],*coeffs[0,:2**(nlab-1)+1]])
    coeffs[1] = np.array([*coeffs[1,2**(nlab-1)+1:],*coeffs[1,:2**(nlab-1)+1]])

    coeffs[0] = np.array([*coeffs[0,-2:],*coeffs[0,:-2]])
    coeffs[1] = np.array([*coeffs[1,-2:],*coeffs[1,:-2]])

    nint1 = get_nint(coeffs[0])
    nint2 = nintx + nint1# - 1
    nint3 = get_nint(coeffs[1])

    npres1 = ncut0-nint1
    npres2 = (nx - nintx) + npres1
    npres3 = ncut1-nint2

    n1 = npres1 + nint1 + 1
    n2 = npres2 + nint2 + 1
    n3 = npres3 + nint3 + 1

    while np.min(A1x)>bin_to_dec('1'+'0'*(n2-3)+'1', nint=nint2-1, phase=phase) and np.max(A1x)<bin_to_dec('0'+'1'*(n2-2)+'1', nint=nint2-1, phase=phase):
        nint2 = nint2 - 1
        n2 = npres2+nint2+1

    nint2 = nint2 + 1
    n2 = npres2 + nint2 + 1

    n = n2
    nc = n1

    nintcs = np.array([[nint1,nint3]])
    nint = nint2

    if 16*(2**(nc+n+nx+nlab))/2**20>7568:
        raise ValueError('Too many qubits!',nc+n+nx+nlab)

    return n, nc, nlab, nint, nintcs, coeffs, bounds

def optimize_coeffs_qubits(f_x, xs, m, npres0, npres1, norder=1, phase=True, label_swap=False):

    xmin = xs[0]
    xmax = xs[-1]

    nx = int(np.log2(len(xs)))
    nintx = get_nint(xs)
    npresx = get_npres(xs)

    if xmin==0.:
        xphase=False
    else:
        xphase=True
    
    bounds_ = np.linspace(xmin, xmax, (2**m)+1)

    bounds__ = []
    for bound in bounds_:
        bounds__.append(bin_to_dec(my_binary_repr(bound, n=nx, nint=nintx, phase=xphase), nint=nintx, phase=xphase))
    bounds_ = bounds__

    coeffs = get_bound_coeffs(f_x, bounds_, norder, reterr=False).T
    bounds = np.copy(np.array(bounds_[:-1]))

    # The number of integer bits of A1, A0 and the result
    nint0 = get_nint(coeffs[0])
    nint1 = get_nint(coeffs[1])
    nint = nintx + nint0

    # The number of precision bits of the result
    npres = npresx + npres0

    # The total number of bits of A1, A0 and the result
    n0 = npres0 + nint0 + 1
    n1 = npres1 + nint1 + 1
    n = npres + nint + 1

    rcoeffs = []
    for coeff in coeffs[0]:
        # Convert coefficient to binary string of new length n0, then calculate the corresponding decimal value
        rcoeff = bin_to_dec(my_binary_repr(coeff, n0, nint=nint0, phase=phase), nint=nint0, phase=phase)
        rcoeffs.append(rcoeff)
    
    coeffs[0] = np.array(rcoeffs)

    # Calculate the differences between the f(x) and A1x over all x
    fdifs = f_x(xs) - piecewise_poly(xs, np.array([coeffs[0],np.zeros(len(coeffs[1]))]).T, bounds)
    coeffs_ = []
    bounds__ = np.copy(bounds_)
    bounds__[-1] = np.inf
    for i in np.arange(len(bounds__))[:-1]:
        # Calculate the mean differences in each domain to be the new bias A0
        coeffs_.append(np.mean(fdifs[np.greater_equal(xs,bounds__[i])&np.greater(bounds__[i+1],xs)]))

    coeffs[1] = np.array(coeffs_)    

    nint1 = get_nint(coeffs[1])

    rcoeffs = []
    for coeff in coeffs[1]:
        # Convert coefficient to binary string of new length n1, then calculate the corresponding decimal value
        rcoeff = bin_to_dec(my_binary_repr(coeff, n1, nint=nint1, phase=phase), nint=nint1, phase=phase)
        rcoeffs.append(rcoeff)

    coeffs[1] = np.array(rcoeffs)
    
    coeffs[0,-1] = 0.
    coeffs[1,-1] = bin_to_dec(my_binary_repr(f_x(bounds[-1]), n1, nint=nint1, phase=phase), nint=nint1, phase=phase)

    # The number of integer bits of A1, A0 and the result
    nint0 = get_nint(coeffs[0])
    nint1 = get_nint(coeffs[1])
    nint = nintx + nint0

    # The number of precision bits of the result
    npres = npresx + npres0

    # The total number of bits of A1, A0 and the result
    n0 = npres0 + nint0 + 1
    n1 = npres1 + nint1 + 1
    n = npres + nint + 1

    ys_rnd = piecewise_poly(xs, coeffs.T, bounds)
    A1x = piecewise_poly(xs, np.array([coeffs[0],np.zeros(len(coeffs[1]))]).T, bounds)

    nint = get_nint([A1x, ys_rnd])
    n = npres + nint + 1

    nintcs = np.array([[nint0,nint1]])

    if 16*(2**(n0+n+nx+m))/2**20>7568:
        raise ValueError('Too many qubits!',n0+n+nx+m)

    if label_swap:
        coeffs[0] = np.array([coeffs[0,-1],*coeffs[0,:-1]])
        coeffs[1] = np.array([coeffs[1,-1],*coeffs[1,:-1]])

    bounds[-1] = xmax

    return n, n0, nint, nintcs, coeffs, bounds


def get_bound_coeffs(func, bounds, norder, reterr=False):
    if np.array(bounds).ndim==0:
        print('Bounds must be a list of two entries!')
    if len(bounds)==1:
        print('Bounds must be a list of two entries!')

    coeffs = []
    errs = []
    for i in np.arange(len(bounds))[:-1]:
        coeffs_, err_ = run_remez(func, bounds[i], bounds[i+1], norder)
        coeffs.append(np.array(coeffs_))
        errs.append(err_)
    if reterr:
        return np.array(coeffs), np.array(errs)
    else:
        return np.array(coeffs)

def _get_chebyshev_nodes(n, a, b):
    nodes = [.5 * (a + b) + .5 * (b - a) * np.cos((2 * k + 1) / (2. * n) * np.pi)
             for k in range(n)]
    return nodes

def _get_errors(exact_values, poly_coeff, nodes):
    ys = np.polyval(poly_coeff, nodes)
    for i in range(len(ys)):
        ys[i] = abs(ys[i] - exact_values[i])
    return ys

def run_remez(fun, a, b, d=5, odd=False, even=False, tol=1.e-13):
    finished = False
    # initial set of points for the interpolation
    cn = _get_chebyshev_nodes(d + 2, a, b)
    # mesh on which we'll evaluate the error
    cn2 = _get_chebyshev_nodes(100 * d, a, b)
    # do at most 50 iterations and cancel if we "lose" an interpolation
    # point
    it = 0
    while not finished and len(cn) == d + 2 and it < 50:
        it += 1
        # set up the linear system of equations for Remez' algorithm
        b = np.array([fun(c) for c in cn])
        A = np.matrix(np.zeros([d + 2,d + 2]))
        for i in range(d + 2):
            x = 1.
            if odd:
                x *= cn[i]
            for j in range(d + 2):
                A[i, j] = x
                x *= cn[i]
                if odd or even:
                    x *= cn[i]
            A[i, -1] = (-1)**(i + 1)
        # this will give us a polynomial interpolation
        res = np.linalg.solve(A, b)

        # add padding for even/odd polynomials
        revlist = reversed(res[0:-1])
        sc_coeff = []
        for c in revlist:
            sc_coeff.append(c)
            if odd or even:
                sc_coeff.append(0)
        if even:
            sc_coeff = sc_coeff[0:-1]
        # evaluate the approximation error
        errs = _get_errors([fun(c) for c in cn2], sc_coeff, cn2)
        maximum_indices = []

        # determine points of locally maximal absolute error
        if errs[0] > errs[1]:
            maximum_indices.append(0)
        for i in range(1, len(errs) - 1):
            if errs[i] > errs[i-1] and errs[i] > errs[i+1]:
                maximum_indices.append(i)
        if errs[-1] > errs[-2]:
            maximum_indices.append(-1)

        # and choose those as new interpolation points
        # if not converged already.
        finished = True
        for idx in maximum_indices[1:]:
            if abs(errs[idx] - errs[maximum_indices[0]]) > tol:
                finished = False

        cn = [cn2[i] for i in maximum_indices]

    return sc_coeff, max(abs(errs))
