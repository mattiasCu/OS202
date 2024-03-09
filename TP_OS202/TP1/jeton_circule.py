from mpi4py import MPI

comm = MPI.COMM_WORLD.Dup()
rank = comm.Get_rank()
size = comm.Get_size()
#size = min(int(input("Nombre de processeurs : ")), comm.Get_size())
#print(size)

token = 1

if rank == 0:
    # Processus de rang zéro
    comm.send(token, dest=1)
    print(f"Processus {rank}: Envoi du jeton {token} au processus 1")
    token = comm.recv(source=size - 1)
    print(f"Processus {rank}: Reçu le jeton {token} du processus {size - 1}")
else:
    # Autres processus
    source = rank - 1
    dest = (rank + 1) % size

    token = comm.recv(source=source)
    print(f"Processus {rank}: Reçu le jeton {token} du processus {source}")

    token += 1
    comm.send(token, dest=dest)
    print(f"Processus {rank}: Incrémenté et envoyé le jeton {token} au processus {dest}")

comm.Barrier()  # Synchronisation de tous les processus

if rank == 0:
    print(f"Processus {rank}: Le jeton final est {token}")


# COMMANDE A UTILISER :
# mpiexec -n nbp python3 jeton_circule.py
