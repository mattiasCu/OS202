from mpi4py import MPI
from time import time
# import random as rd
import numpy as np



def fusion(t1,t2):
    tab = []
    n1=len(t1)
    n2=len(t2)

    k1=0
    k2=0
    while k1<n1 and k2<n2:
        if t1[k1]<t2[k2]:
            tab.append(t1[k1])
            k1+=1
        else:
            tab.append(t2[k2])
            k2+=1
    
    while k1<n1:
        tab.append(t1[k1])
        k1+=1
    while k2<n2:
        tab.append(t2[k2])
        k2+=1
    return tab

def fusion_sort(tab):
    n = len(tab)
    if len(tab)==1:
        return tab
    else:
        t1 = fusion_sort(tab[:n//2])
        t2 = fusion_sort(tab[n//2:])
        return fusion(t1,t2)

# =========== Programme ===========

comm = MPI.COMM_WORLD.Dup()
rank = comm.Get_rank()
size = comm.Get_size()

nbBuck = size
buckets = [[] for _ in range(nbBuck)]




if rank == 0:
    # Processus de rang zéro
    
    # tab = [rd.uniform(0.0,100.0) for _ in range(10000)]
    #tab = [4,5,6,2,7,9,1,-1,-4,-5]
    tab = np.random.rand(10000)

    deb = time()
    tmax = np.max(tab)
    tmin = np.min(tab)
    interval = (tmax-tmin)/nbBuck # on suppose une répartition équilibrée
    
    for k in range(len(tab)):
        buckets[min(int((tab[k]-tmin)/interval), nbBuck-1)].append(tab[k])
    buckets = [np.array(k) for k in buckets]

bucket = comm.scatter(buckets, root = 0)

bucket = np.sort(bucket)
print(bucket[0:5])
print(bucket.size)

buckets = comm.gather(bucket, root = 0)

if rank == 0:
    tests = [k[0] for k in buckets]
    indexed_list = [(value, index) for index, value in enumerate(tests)]
    sorted_indices = [index for value, index in sorted(indexed_list)]

    sortedTab = buckets[sorted_indices[0]]
    for k in sorted_indices[1:]:
        np.append(sortedTab,buckets[k])
    
    fin = time()
    print(f"Temps du tri bucket : {fin-deb}")
    for k in range(len(sortedTab)-1):
        if sortedTab[k]>sortedTab[k+1]:
            print("issue")
    print(sortedTab)

# else:
#     # Autres processus

#     token = comm.recv(0)
#     print(f"Processus {rank}: Reçu le jeton {token} du processus {source}")

#     token += 1
#     comm.send(token, dest=dest)
#     print(f"Processus {rank}: Incrémenté et envoyé le jeton {token} au processus {dest}")

# comm.Barrier()  # Synchronisation de tous les processus



#mpiexec / mpirun