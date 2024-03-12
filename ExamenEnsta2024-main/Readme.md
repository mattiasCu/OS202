L'ordinateur a 8 coeurs de calcul, 131072 Bytes = 128 kB iL1 cachet, 65536 Bytes = 64 kB dL1 cache, 4194304 Bytes = 4096 kB = 2 MB L2 cache


Part 1

Nous parallélisons ce programme en divisant la matrice du graphe par le nombre de cœurs. Ensuite, nous effectuons les calculs sur chaque processus et les illustrons sur un même graphique.

Il est apparent que des frontières nettes existent dans ce graphique, résultat de la méthode de parallélisation : les limites du graphique coloré n'ont pas été explicitement traitées, donc chaque section du graphique agit comme un graphe indépendant, d'où l'apparition des frontières. De plus, la découpe de l'image indiquée va affecter l'algorithme de gradient conjugué, car cette image ne sera pas complète dans ce contexte.

Part 2
