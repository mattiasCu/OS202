from mpi4py import MPI

comm = MPI.COMM_WORLD.Dup()
rank = comm.Get_rank()
size = comm.Get_size()
#size = min(int(input("Nombre de processeurs : ")), comm.Get_size())
#print(size)

token = 1

if rank == 0:
    # Processus de rang zéro
    token = int(input("Veuillez entrer un nombre : "))
    #On suppose qu'il y a 2^d processus
    k = 0
    while (2**k < size):
        print(f"Processus {rank}: envoi à  {rank + 2**k}")
        comm.send(token, dest=rank + 2**k)
        k+=1
    print(f"Processus {rank}: l'entier est {token}")

elif rank ==1: 
    token = comm.recv(source=0)
    k=1
    while (rank + 2**k < size):
        print(f"Processus {rank}: envoi à  {rank + 2**k}")
        comm.send(token, dest=rank + 2**k)
        k+=1
    print(f"Processus {rank}: l'entier est {token}")

else:
    # Autres processus
    k=0
    while 2**k < rank:
        k+=1
    if 2**k != rank :
        k-=1

    token = comm.recv(source=rank-2**k)
    
    while (rank + 2**k < size):
        print(f"Processus {rank}: envoi à  {rank + 2**k}")
        comm.send(token, dest=rank + 2**k)
        k+=1
    print(f"Processus {rank}: l'entier est {token}")


comm.Barrier()  # Synchronisation de tous les processus
