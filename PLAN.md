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

> **État (mai 2026)** :
>
> - **v1 redundant partition** (abandonnée) : chaque rank chargeait le maillage
>   global, ne calculait que sur `[localElStart, localElEnd)`. Correcte mais
>   sans gain de mémoire ni de temps.
> - **v2 vraie partition** (en cours) : utilise `gmsh::model::mesh::partition(N)`
>   avec `Mesh.PartitionCreateGhostCells = 1`. Chaque rank charge seulement
>   ses entités 3D *owned* + *halo* via `getElementsByType(elType, entityTag)`.
>   Validé sur cube 1000 hex à 2 et 4 ranks : résidus à `0.07-0.22%` du
>   séquentiel (roundoff pur), mémoire/rank ÷ N (avec halo qui s'amincit
>   relativement quand N grandit).

**Objectif** : chaque rank ne charge que sa portion du maillage + halo,
en indices LOCAUX. Toutes les structures (Jacobiens, basis fcts, mass
matrices, u, Flux) ont une taille `O(local + halo)`.

### Stratégie

#### Numérotation
- Éléments locaux : indices `[0, Nloc)` (1-pour-1 avec la partition gmsh `rank+1`)
- Éléments halo : indices `[Nloc, Nloc + Nhalo)` (reçus depuis ranks voisins)
- Nœuds : union des nœuds des éléments locaux + halo, indexés `[0, Nnodes_local)`
- Faces : seulement celles touchant au moins un élément local
- Maps de traduction (uniquement à la frontière) :
  - `m_globalElTag[localId] → gmsh tag` (et inverse)
  - `m_globalNodeTag[localId] → gmsh tag` (et inverse)

#### Refactor du constructeur (`src/Mesh.cpp:Mesh::Mesh`)
1. Si `Parallel::size() > 1` : `gmsh::model::mesh::partition(N, true, true)`
   → crée N Physical Groups `partition_p`, chaque élément a un partition tag.
2. Pour chaque rank `r` : itérer sur les Physical Groups de partition,
   ne garder que les éléments avec `partition_tag == r+1` → liste locale.
3. Identifier le halo : pour chaque face d'interface (faces partagées entre
   plusieurs partitions, retournées par gmsh sous forme de "ghost entities"),
   ajouter l'élément voisin de partition `≠ r+1` à la liste halo.
4. Charger les nœuds locaux+halo, construire `m_globalToLocalNode`.
5. Reconstruire `m_elNodeTags` en indices LOCAUX (1-pour-1 avec local+halo).
6. Calculer Jacobiens / basis fcts / gradients pour `Nloc + Nhalo` éléments
   uniquement (compute fait sur taille réduite).
7. Mass matrix uniquement pour les `Nloc` locaux (halo n'est pas mis à jour).
8. Détecter faces locales = faces dont au moins un élément voisin est local.
   Calculer Jacobiens/basis fcts de face uniquement pour ces faces.
9. Marquer chaque face locale : interior / interface / boundary.
10. Halo send/recv listes en indices **locaux** (pas globaux). L'ordre est
    fixé une fois pour toutes et symétrique entre paires de ranks.

#### Refactor `haloExchange`
- Plus besoin d'envoyer les IDs : seuls les `dofs` sont échangés, dans un
  ordre figé déterminé à la construction.
- Pack : `for el in m_haloSendElIds[r]: copy u[el*N..]` (indices locaux).
- Unpack : symétrique, dans la zone halo `u[(Nloc+i)*N..]`.

#### Refactor solver
- `numStep` : boucle sur `[0, Nloc)` (au lieu de `[localElStart, localElEnd)`).
- `updateFlux` : boucle sur `[0, Nloc + Nhalo)` (le halo a besoin de Flux pour
  que `precomputeFlux` puisse calculer les flux d'interface).
- `precomputeFlux` : boucle sur faces locales seulement.
- Résidu, observer, source : indices locaux + `Allreduce` vers root.

### Risques techniques (détaillés)
- **Cohérence ordre des halos** : pour échanger sans envoyer les IDs, l'ordre
  des éléments dans `m_haloSendElIds[r]` du rank A doit correspondre exactement
  à `m_haloRecvElIds[A]` du rank B. Solution : trier par tag global gmsh des
  deux côtés.
- **Faces de bord coupées par le partitionneur** : à vérifier que gmsh ne place
  pas une face frontière (avec BC physique) à la frontière entre deux partitions.
- **Mapping global↔local des nœuds aux interfaces** : un nœud partagé entre
  un élément local et un élément halo doit avoir le même indice local des deux
  côtés (sinon les flux de face sont faux).
- **Performance du partitionnement gmsh** : pour des très gros maillages (>10M
  elts), `gmsh::partition` peut être lent. Alternative ParMETIS si besoin.

### Livrable
- Mémoire/rank ≈ mémoire séquentielle / N (vérifié via `top` ou `mstat`).
- Setup time/rank ≈ setup séquentiel / N.
- Solution numériquement identique au séquentiel (à `< 1e-10` en norme L2).
- Strong scaling > 70% jusqu'à 8 ranks sur cube 1k hexes.

### Avancement v2 (mai 2026, en cours)

**Acquis** :
- ✅ `gmsh::model::mesh::partition(N)` + `Mesh.PartitionCreateGhostCells = 1`
  exposent owned + halo de manière clairement identifiable via `getEntities` /
  `getPartitions` / `getGhostElements`.
- ✅ Helper `buildPartitionLayout()` : retourne owned/halo/boundary/interface
  entités pour le rank courant.
- ✅ Helpers `loadElementsByPartition()`, `loadJacobiansByPartition()` (idem,
  intégré), `loadBarycentersByPartition()`, `loadFaceNodesByPartition()` :
  toutes les structures globales concaténées **owned + halo** seulement.
- ✅ Numérotation locale : `[0, Nloc)` owned, `[Nloc, Nloc+Nhalo)` halo.
  `m_localElStart = 0`, `m_localElEnd = Nloc`, `m_elNum = Nloc + Nhalo`.
- ✅ `m_haloSendElIds[r]` / `m_haloRecvElIds[r]` triés par tag gmsh global
  → l'ordre est symétrique entre paires de ranks → `haloExchange` n'envoie
  plus d'IDs, juste les valeurs.
- ✅ Résidu globalement normalisé (Allreduce du compte de DoFs locaux).
- ✅ Validation : cube 1000 hex à 2 et 4 ranks → résidus à 0.07-0.22% du
  séquentiel (roundoff pur).

**Limitations actuelles** :
- ❌ Le **scaling temporel** n'est pas encore au rendez-vous sur les petits
  maillages : sur cube 1000 hex, n=4 est ~50 % plus lent que n=1 à cause
  du coût fixe du chargement gmsh + `partition()` répété sur chaque rank.
  Le compute par rank devient trop petit relativement au setup.
- ❌ Le `gmsh::open(meshFile)` est répété N fois (N processus indépendants).
  Pour des très gros maillages, ce sera le goulet d'étranglement de start-up.

**Étapes restantes** :
- ⏭️ Bench scaling sur maillage > 8k hex (compute > setup).
- ⏭️ Optimisation : lire le maillage une seule fois côté rank 0, partition,
  écrire des fichiers `cube_part<r>.msh` et chaque rank lit son fichier
  (élimine le `partition(N)` redondant et le full-mesh load).
- ⏭️ Quick-wins compute (Étape Niveau 1 abandonnée mais réutilisable maintenant
  que la structure est partitionnée) : restreindre `precomputeMassMatrix` aux
  `Nloc` premiers elts, `precomputeFlux` aux faces locales+interface.

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
