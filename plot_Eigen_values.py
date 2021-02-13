import os.path
import sys
import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    print("Error importing pyplot from matplotlib, please install matplotlib package first...")
    sys.tracebacklimit=0
    raise Exception("Importing matplotlib failed")

if __name__ == '__main__':

    #b = np.loadtxt('/home/liuq0a/tests/marmousi_onshore_SRVM_new10_1/output.stats/misfit')
    b = np.loadtxt('Eigen_values.txt')
    b0 = b[0]
    #b = b / b[0]

    print "b = ", b[:]

    nb = np.size(b)

    plt.figure(figsize=(10,5))
    ax = plt.subplot(111)
    plt.plot(range(1,nb+1),b,'o-',color="black",linewidth=2)
    #plt.plot(range(na),a,label='$L-BFGS$',color="blue",linewidth=2)
    ax.legend()
    ax.set_yscale('log')
    plt.xlabel("ORDER")
    plt.ylabel("EIGENVALUE")
    plt.xlim(0,70)
    #plt.ylim(5.0e-3,1.1)
    #plt.title("MARMOUSI")
    plt.savefig("fig3.png")

