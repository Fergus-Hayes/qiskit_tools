import qiskit_tools as qt
import numpy as np

# weights we need to round
As = np.random.uniform(-5,5,10)

# number of bits to store the integers
nint = qt.get_nint(As)

# number of bits to store the fraction
print('Precision bits:',qt.get_npres(As))

npres = 5
n = nint + npres + 1

print(As)

Bs = []
for A in As:
    Bs.append(qt.bin_to_dec(qt.my_binary_repr(A, n, nint=nint, phase=True), nint=nint, phase=True))

print(Bs)
