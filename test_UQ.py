import numpy as np
import scipy.linalg
import io,os
import subprocess
import errno    



class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def mkdir_p(path):
	try:
		os.makedirs(path)
	except OSError as exc:
		if exc.errno == errno.EEXIST and os.path.isdir(path):
		    pass
		else:
		    raise

import sys
def apply(q, S, Y, LBFGS_mem):

	kk = LBFGS_mem
	rh = np.zeros(kk)
	al = np.zeros(kk)
	for ii in range(kk):
		rh[ii] = 1/np.dot(Y[:,ii], S[:,ii])
		al[ii] = rh[ii]*np.dot(S[:,ii], q)
		q = q - al[ii]*Y[:,ii]

	r = q * np.load("precond0.npy")

	sty = np.dot(Y[:,0], S[:,0])
	yty = np.dot(Y[:,0], Y[:,0])

	for ii in range(kk-1, -1, -1):
		be = rh[ii]*np.dot(Y[:,ii], r)
		r = r + S[:,ii]*(al[ii] - be)

	return r

m = len(np.load('g_new'))

S=[]
Y=[]

LBFGS_mem = 100
LBFGS_mem_used = 49

R = np.random.randn(m, LBFGS_mem_used)
r = np.zeros((m, LBFGS_mem_used))

n = LBFGS_mem
S = np.memmap('LBFGS/S', mode='r', dtype='float32', shape=(m, n))
Y = np.memmap('LBFGS/Y', mode='r', dtype='float32', shape=(m, n))

for ii in range(LBFGS_mem_used):
	print ii
	r[:,ii] = apply(R[:,ii], S, Y, LBFGS_mem_used)
	r[:,ii] = r[:,ii] - R[:,ii] * np.load("precond0.npy")


Q, _ = np.linalg.qr(r)

Zt = np.dot(R.transpose(),Q)

Wt = np.dot(r.transpose(),Q)

B = np.linalg.solve(Zt,Wt)

Bt = 0.5 * (B + B.transpose())

U, S, _ = np.linalg.svd(Bt, full_matrices=False)

U = np.dot(Q,U)

print np.shape(S)

S = S / 4.957558e+04

np.savetxt('Eigen_values.txt',S)

S_t = np.diag(S)
mkdir_p('./UQ_map')

U_t = np.dot(U, S_t)
UQ_map = np.zeros(m)

for ii in range(m):
	UQ_map[ii] = np.inner(U[ii,:],U_t[ii,:])

UQ_map = np.sqrt(UQ_map)

file_vp = './UQ_map' + '/proc000000_vp.bin'
file_vs = './UQ_map' + '/proc000000_vs.bin'
UQ_map[0:m/2].real.astype('float32').tofile(file_vp)
UQ_map[m/2:].real.astype('float32').tofile(file_vs)

with cd("./UQ_map"):
	subprocess.call("./pre_plot")

sys.exit(0)
for ii in range(LBFGS_mem_used):
	print ii
	snum = "{:03d}".format(ii)
	mkdir_p(snum)
	cmd1 = "cp ./pre_plot ./proc000000_x.bin ./proc000000_z.bin " + snum
	os.system(cmd1)
	cmd1 = "cp ./ps.sh ./plot_kernel* " + snum
	os.system(cmd1)
	file_vp = snum + '/proc000000_vp.bin'	
	file_vs = snum + '/proc000000_vs.bin'
	U[0:m/2,ii].real.astype('float32').tofile(file_vp)
	U[m/2:,ii].real.astype('float32').tofile(file_vs)
	with cd(snum):
		subprocess.call("./pre_plot")
	cd("../")




