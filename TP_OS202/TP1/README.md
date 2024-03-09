
# TD1

`pandoc -s --toc README.md --css=./github-pandoc.css -o README.html`





## lscpu

```
coller ici les infos *utiles* de lscpu. 
```

*Des infos utiles s'y trouvent : nb core, taille de cache*
Processeur(s) :                             12
  Liste de processeur(s) en ligne :         0-11
Caches (sum of all):                        
  L1d:                                      192 KiB (6 instances)
  L1i:                                      192 KiB (6 instances)
  L2:                                       3 MiB (6 instances)
  L3:                                       8 MiB (2 instances)
    Thread(s) par cœur :                    2
    Cœur(s) par socket :                    6



## Produit matrice-matrice



### Permutation des boucles

*Expliquer comment est compilé le code (ligne de make ou de gcc) : on aura besoin de savoir l'optim, les paramètres, etc. Par exemple :*

`make TestProduct.exe && ./TestProduct.exe 1024`
-O3 : force le compilateur à optimiser au maximum le programme.
-march=native : applique des optimisations qui sont propres au CPU de l'appareil.
-frecord-gcc-switches : enregistre les informations de compilation dans le fichier objet créé

  ordre           | time    | MFlops  | MFlops(n=1024) 
------------------|---------|---------|----------------
i,j,k (origine)   | 2.73764 | 782.476 |   181          
j,i,k             |  |  |                 211
i,k,j             |  |  |                 73
k,i,j             |  |  |                 58
j,k,i             |  |  |                 5474
k,j,i             |  |  |                 2488




*Discussion des résultats*
Lorsqu'un élément d'une matrice est utilisé, les 8 suivants sont mis en cache (ceux qui sont en-dessous dans la colonne).
Ainsi, ces éléments sont déjà en cache lorsqu'ils sont utilisés si l'on parcourt les boucles ligne par ligne plutôt que colonne par colonne.
NB : 1024 est le modulo du cache, ce qui ralentit beaucoup les calculs

### OMP sur la meilleure boucle 

`make TestProduct.exe && OMP_NUM_THREADS=8 ./TestProduct.exe 1024`

  OMP_NUM         | MFlops  | MFlops(n=2048) | MFlops(n=512)  | MFlops(n=4096)
------------------|---------|----------------|----------------|---------------
1                 |  |          4869            12628           5232
2                 |  |          9271            22444           9013
3                 |  |          13501           28238           12567
4                 |  |          15922           35452           15595
5                 |  |          16258           35444           15297
6                 |  |          17808           44481           16282
7                 |  |          23794           35122           20556
8                 |  |          27856           36628           24694

Il faudrait que les éléments des matrices utilisés aient été dans le cache à chaque fois qu'ils ont été utilisés pour optimiser le calcul.
On peut le faire en faisant des calculs par blocs


### Produit par blocs

`make TestProduct.exe && ./TestProduct.exe 1024`

  szBlock         | MFlops  | MFlops(n=2048) | MFlops(n=512)  | MFlops(n=4096)
------------------|---------|----------------|----------------|---------------
origine (=max)    |  |          5440            12663           5253
32                |  |          6617            7341            6429
64                |  |          9138            8976            8310
128               |  |          9787            9767            9025
256               |  |          10484           10026           6653
512               |  |          6388            11990           4206
1024              |  |          4243            failed...       4147




### Bloc + OMP



  szBlock      | OMP_NUM | MFlops  | MFlops(n=2048) | MFlops(n=512)  | MFlops(n=4096)|
---------------|---------|---------|-------------------------------------------------|
A.nbCols       |  1      |         |      4880          |    12741            |    3629           |
512            |  8      |         |      30590          |   36276             |   22524            |
---------------|---------|---------|-------------------------------------------------|
Speed-up       |         |         |      0.912          |   0.991             |   1.1            |
---------------|---------|---------|-------------------------------------------------|

Il est plus difficile de paralléliser l'algorithme par blocs que le naïf, ce qui compense le gain en performances

### Comparaison with BLAS

L'algorithme blas donne un nombre de FLOPS constant, il n'est pas parallélisable et finit par être moins efficace que les autres algorithmes.
Le meilleur algorithme dépend des ressources disponibles :
l'algo naïf remporte si l'on a de nombreux threads disponibles,
si on en a qu'un, le produit par blocs est meilleur.
Au final, l'algorithme le plus optimisé ne l'est pas car il n'exploite pas les threads et la mise en cache?


# Tips 

```
	env 
	OMP_NUM_THREADS=4 ./produitMatriceMatrice.exe
```

```
    $ for i in $(seq 1 4); do elap=$(OMP_NUM_THREADS=$i ./TestProductOmp.exe|grep "Temps CPU"|cut -d " " -f 7); echo -e "$i\t$elap"; done > timers.out
```

### Calcul de pi

Avec 1 processus : 0.900s
Avec 5 processus (dont 4 qui calculent) : 0.327s
speedup = 2,75
nb de processus choisi manuellement.

### hypercube