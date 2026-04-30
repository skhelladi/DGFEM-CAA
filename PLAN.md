# Plan de parallélisation MPI massive — DGFEM-CAA

## Contexte

Le solveur DGFEM-CAA est actuellement parallélisé uniquement via OpenMP (mémoire partagée), limitant son passage à l'échelle à un seul nœud de calcul. L'objectif est d'introduire une parallélisation MPI par **décomposition de domaine** afin de :

- Distribuer le maillage sur plusieurs nœuds (clusters HPC)
- Traiter des cas industriels (millions d'éléments) inaccessibles en mémoire partagée
- Conserver le mode hybride **MPI + OpenMP** (MPI inter-nœuds, OpenMP intra-nœud)
- Préserver la précision numérique (les flux numériques DG sont locaux par nature → bien adaptés à MPI)

Le code Galerkin Discontinu se prête naturellement à la décomposition de domaine : les volumes intégrés sont strictement locaux par élément, et le couplage entre éléments se fait uniquement via les flux numériques sur les faces (Rusanov). Une couche de **halo** (éléments fantômes) suffit pour assembler les flux à l'interface entre sous-domaines.

---

## Architecture cible

```
┌─────────────────────────────────────────────────────────┐
│  MPI rank i  (1 sous-domaine)                           │
│  ┌──────────────────────────┐  ┌─────────────────────┐  │
│  │ Éléments locaux          │  │ Halo (ghost cells)  │  │
│  │ - calcul volumique       │  │ - reçus des voisins │  │
│  │ - mise à jour solution   │  │ - lecture seule     │  │
│  └──────────────────────────┘  └─────────────────────┘  │
│              ↕ OpenMP intra-rank (déjà en place)        │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Faces internes / Faces interfaces / Faces bords  │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────↕ MPI ─────────────────────────────┘
              Halo exchange à chaque étape RK
```

---

## Phase 1 — Préparation et infrastructure (1-2 semaines)

**Objectif** : intégrer MPI dans le build sans modifier la logique, valider l'environnement.

### Tâches
1. **Build system** (`CMakeLists.txt`, `build.sh`)
   - Ajouter `find_package(MPI REQUIRED)` et `target_link_libraries(... MPI::MPI_CXX)`
   - Installer `openmpi` ou `mpich` (Linux : apt ; macOS : `brew install open-mpi`)
   - Option CMake `DG_USE_MPI=ON/OFF` pour conserver une compilation séquentielle

2. **Initialisation MPI** (`src/dgalerkin.cpp:34`)
   - `MPI_Init_thread(MPI_THREAD_FUNNELED)` (compatible OpenMP)
   - Récupérer `rank` et `size` ; `MPI_Finalize` avant `gmsh::finalize()`
   - Wrapper `Logger` filtrant les sorties au rank 0 uniquement

3. **Couche d'abstraction** (nouveau fichier `include/Parallel.h` + `src/Parallel.cpp`)
   - Singleton léger : `Parallel::rank()`, `Parallel::size()`, `Parallel::comm()`
   - Helpers : `allReduce`, `barrier`, `bcast` typés

### Livrable
Un exécutable `mpirun -n N ./dgalerkin config.json` qui tourne en N copies indépendantes (chaque rank lit le maillage entier et fait le calcul redondant). Aucun gain de perf, mais l'infra MPI est validée.

---

## Phase 2 — Partitionnement du maillage (2-3 semaines)

**Objectif** : chaque rank ne charge que sa portion du maillage + halo.

### Tâches
1. **Utiliser le partitionneur natif de gmsh**
   - `gmsh::model::mesh::partition(N)` génère N partitions équilibrées (METIS interne)
   - Chaque partition devient une `Physical Group` ; chaque rank lit uniquement sa partition via les API gmsh
   - Alternative : appeler METIS/ParMETIS directement si on veut un contrôle fin

2. **Refactorer `Mesh::Mesh()`** (`src/Mesh.cpp:20`)
   - Lire uniquement les éléments de la partition `rank`
   - Construire la table de **correspondance globale ↔ locale** (nodeTags, elTags)
   - Identifier 3 catégories de faces :
     - **Internes** : entre deux éléments locaux → traitement actuel
     - **Interface** : entre élément local et élément distant → halo exchange
     - **Bord physique** : boundary condition (existant)

3. **Construction du halo**
   - Pour chaque interface, identifier le rank voisin et l'élément distant
   - Stocker dans `Mesh` :
     - `m_haloSendElements[rank]` : éléments locaux à envoyer
     - `m_haloRecvElements[rank]` : éléments fantômes à recevoir
     - `m_haloFaces` : faces couplées à un halo
   - Allouer le buffer fantôme pour `u[4]` étendu : `[localNodes + haloNodes]`

4. **Tests**
   - Cas simple : mesh 2D divisé en 2/4 partitions → vérifier que la somme des éléments locaux = total global
   - Reproduire un calcul séquentiel et un calcul à 2 ranks sans halo exchange (résultats faux mais code stable)

### Livrable
Un maillage partitionné chargé correctement. La table de connectivité halo est prête mais pas encore utilisée dans le solveur.

---

## Phase 3 — Échanges halo et flux numériques distribués (2-3 semaines)

**Objectif** : les flux Rusanov à l'interface sont corrects → résultats numériquement identiques au séquentiel.

### Tâches
1. **Implémenter `Mesh::haloExchange(u)`**
   - Pack des données : pour chaque rank voisin, copier `u[0..3][haloSendElements]` dans un buffer contigu
   - `MPI_Isend` / `MPI_Irecv` non-bloquants (un échange par paire de ranks)
   - `MPI_Waitall` puis unpack dans la zone halo de `u`
   - **Critique** : ordonner les nœuds halo de manière cohérente entre ranks (utiliser tags globaux gmsh)

2. **Modifier la boucle temporelle** (`src/solver.cpp:413-573` pour RK4)
   - Avant chaque étape RK : `mesh.haloExchange(u_stage)`
   - Avant `mesh.updateFlux()` : déclencher l'échange (overlap calcul/comm si possible avec `MPI_Isend`/`MPI_Irecv` lancés tôt)
   - `Mesh::precomputeFlux()` (`src/Mesh.cpp:761-813`) : pour les faces d'interface, utiliser les valeurs halo au lieu de `m_fNbrElIds`

3. **Recouvrement calcul/communication** (optimisation)
   - Découper la boucle des éléments en deux : **éléments intérieurs** (sans face d'interface) et **éléments de bord de domaine** (avec face d'interface)
   - Lancer `MPI_Isend`/`MPI_Irecv` → calculer flux des faces internes pendant la comm → `MPI_Waitall` → calculer flux des faces d'interface
   - Garder le mode "synchrone simple" en option pour debug

4. **Validation rigoureuse**
   - Comparer la solution à `t = T_final` entre runs séquentiel (1 rank) et parallèle (4, 8, 16 ranks) sur un cas analytique (onde gaussienne)
   - Tolérance : erreur < 1e-12 en norme L2 (les opérations doivent être bit-à-bit identiques modulo l'ordre des sommations sur faces partagées)

### Livrable
Solveur MPI fonctionnel avec résultats équivalents au séquentiel. Pas encore de tuning perf.

---

## Phase 4 — Sorties parallèles et observers (1-2 semaines)

**Objectif** : I/O scalable, sans goulet au rank 0.

### Tâches
1. **VTU/PVD parallèle** (`src/Mesh.cpp:1136`, `1232`)
   - Chaque rank écrit son propre fichier `results_t<i>_rank<r>.vtu`
   - Le rank 0 écrit le `.pvtu` (parallel VTU) pointant vers tous les `.vtu` du timestep
   - Le `.pvd` agrège les `.pvtu` (au lieu des `.vtu`)
   - VTK supporte nativement `vtkXMLPUnstructuredGridWriter`

2. **Observers** (`src/solver.cpp:290, 577`)
   - Pour chaque observer, déterminer quel rank possède le point physique (via gmsh `getElementByCoordinates`)
   - Le rank propriétaire évalue, envoie au rank 0 par `MPI_Gather` au pas de sortie
   - Le rank 0 écrit CSV/WAV (existant)

3. **Sources**
   - Identique aux observers : seul le rank propriétaire applique la source

### Livrable
Sorties paraview lisibles directement, fichiers WAV inchangés du point de vue utilisateur.

---

## Phase 5 — Optimisation et passage à l'échelle (2 semaines)

**Objectif** : efficacité parallèle > 80% jusqu'à 256 cores.

### Tâches
1. **Profilage**
   - `mpiP`, `Score-P` ou `TAU` pour identifier les goulets MPI
   - Mesurer ratio compute/comm, déséquilibre de charge

2. **Optimisations probables**
   - Fusionner les buffers halo (un unique `MPI_Isend` par voisin au lieu de 4 — un par champ)
   - Persistent communications (`MPI_Send_init`/`MPI_Recv_init`) si la topologie est statique
   - Affinité processeur : `--bind-to core --map-by socket`
   - Test du mode hybride MPI + OpenMP (1 rank par socket × N threads)

3. **Étude de scaling**
   - Strong scaling : 1 → 4 → 16 → 64 → 256 ranks sur cas fixe
   - Weak scaling : taille proportionnelle au nombre de ranks
   - Documenter dans `docs/SCALING.md` avec courbes

### Livrable
Rapport de scaling + recommandations d'usage (ranks/cores/threads optimaux).

---

## Phase 6 — Robustesse et documentation (1 semaine)

### Tâches
- Mise à jour `README.md` : section MPI (compilation, exécution, exemples)
- Cas tests CI : 1, 2, 4 ranks sur petit maillage
- Documentation API `Parallel.h`
- Vérification mémoire : `valgrind --tool=memcheck` + `MUST` (MPI correctness checker)

---

## Fichiers critiques à modifier

| Fichier | Modifications |
|---|---|
| `CMakeLists.txt` | `find_package(MPI)`, lien `MPI::MPI_CXX` |
| `build.sh` | Installation `open-mpi` (brew) / `libopenmpi-dev` (apt) |
| `src/dgalerkin.cpp:34` | `MPI_Init_thread`, `MPI_Finalize` |
| `include/Mesh.h:38` | Champs halo : `m_haloSendElements`, `m_haloRecvElements`, `m_haloFaces`, méthode `haloExchange()` |
| `src/Mesh.cpp:20` | Constructeur : lecture partitionnée, mapping global/local |
| `src/Mesh.cpp:761-813` | `precomputeFlux` : utiliser valeurs halo aux interfaces |
| `src/Mesh.cpp:1136, 1232` | I/O : `.pvtu`/`.pvd` parallèle |
| `src/solver.cpp:413-573` | Boucle RK : insérer `haloExchange` à chaque étape |
| `src/solver.cpp:290, 577` | Observers : `MPI_Gather` |
| **Nouveau** `include/Parallel.h`, `src/Parallel.cpp` | Couche d'abstraction MPI |

---

## Vérification end-to-end

1. **Test unitaire de partitionnement** : `mpirun -n 4 ./test_partition mesh.msh` vérifie que la somme des éléments locaux = total global, et que chaque face d'interface a un homologue dans le rank voisin.

2. **Test de régression numérique** : cas onde gaussienne 3D dans cube
   - Run séquentiel : `./dgalerkin gauss.json` → `results_seq.pvd`
   - Run MPI : `mpirun -n 8 ./dgalerkin gauss.json` → `results_mpi.pvd`
   - Comparaison L2 entre les deux solutions à `t_final` : doit être < 1e-12

3. **Test de scaling** : cas réaliste (~1M éléments)
   - 1, 2, 4, 8, 16, 32, 64 ranks → mesurer temps total et tracer accélération
   - Critère : efficacité > 70% à 64 ranks

4. **Validation hybride** : `OMP_NUM_THREADS=4 mpirun -n 8 ./dgalerkin gauss.json` doit donner les mêmes résultats que la version pure MPI.

5. **Cas industriel CI** : ajouter à la suite de tests une exécution `mpirun -n 4` du cas le plus complexe existant pour détecter les régressions.

---

## Estimation totale : 9-13 semaines

| Phase | Durée | Risque |
|---|---|---|
| 1. Infra MPI | 1-2 sem | Faible |
| 2. Partitionnement | 2-3 sem | Moyen (gmsh API) |
| 3. Halo & flux | 2-3 sem | **Élevé** (cœur du travail) |
| 4. I/O parallèle | 1-2 sem | Faible |
| 5. Optimisation | 2 sem | Moyen |
| 6. Robustesse | 1 sem | Faible |

**Risques techniques principaux** :
- Cohérence d'ordre des nœuds aux interfaces (bug subtil source d'erreurs numériques)
- Gestion des conditions limites sur faces partitionnées (cas particulier : face de bord coupée par le partitionneur — à vérifier que gmsh ne le fait pas)
- Déséquilibre de charge si éléments d'ordre élevé concentrés dans un sous-domaine (ParMETIS avec poids peut résoudre)
