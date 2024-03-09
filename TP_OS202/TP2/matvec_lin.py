# Produit matrice-vecteur v = A.u
import numpy as np
from mpi4py import MPI

#Parallélisation
comm = MPI.COMM_WORLD.Dup()
rank = comm.Get_rank()
size = comm.Get_size()

# Dimension du problème (peut-être changé)
dim = 120
slice = int(dim/size)

Aloc = np.array([[(i+j) % dim+1. for i in range(dim)] for j in range(rank*slice, (rank+1)*slice)])
u = np.array([i+1. for i in range(dim)])
v1 = Aloc.dot(u)
for dest in range(size):
    if dest != rank:
        comm.send(v1, dest=dest)

for source in range(size):
    if source == 0 :
        if source == rank:
            v = v1
        else:
            other_part = comm.recv(source=source)
            v = other_part
    else:
        if source == rank:
            v = np.hstack([v,v1])
        else:
            other_part = comm.recv(source=source)
            v = np.hstack([v,other_part])

print(f"v = {v}")

# # Initialisation de la matrice
# A = np.array([[(i+j) % dim+1. for i in range(dim)] for j in range(dim)])
# print(f"A = {A}")

# # Initialisation du vecteur u
# u = np.array([i+1. for i in range(dim)])
# print(f"u = {u}")

# # Produit matrice-vecteur
# v = A.dot(u)
# print(f"v = {v}")

#ça marche!