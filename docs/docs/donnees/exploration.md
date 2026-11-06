# Exploration

Notebook : `notebooks/01.01-kl-data-exploration.ipynb`

Cette étape est **strictement descriptive** : aucun redimensionnement, aucune normalisation et aucun apprentissage. Elle établit un état des lieux reproductible des données brutes.

## Méthode

1. **Inventaire** : `build_inventory` crée une table avec un chemin par image et une étiquette explicite.
2. **Lecture complète** : chaque JPEG est entièrement lu afin de détecter les fichiers corrompus et de relever ses dimensions, son mode et son format.
3. **Tailles de fichiers** : comparaison des tailles sur disque par sous-ensemble.
4. **Empreintes** : `add_file_hash` calcule un SHA-256 par fichier pour repérer les doublons binaires.
5. **Inspection visuelle** : échantillon d'images non annotées, puis affichage exhaustif des images annotées.

## Résultats

| Sous-ensemble | Images | Dimensions | Mode | Format | Erreurs de lecture |
|---|---:|---|---|---|---:|
| `cancer` | 50 | 512 × 512 | RGB | JPEG | 0 |
| `normal` | 50 | 512 × 512 | RGB | JPEG | 0 |
| `sans_label` | 1 406 | 512 × 512 | RGB | JPEG | 0 |
| **Total** | **1 506** | | | | **0** |

### Taille des fichiers (KiB)

| Sous-ensemble | Minimum | Médiane | Moyenne | Maximum |
|---|---:|---:|---:|---:|
| `cancer` | 15,8 | 23,0 | 22,5 | 31,8 |
| `normal` | 17,4 | 25,3 | 29,1 | 64,3 |
| `sans_label` | 13,1 | 23,2 | 24,1 | 73,7 |

![Distribution des tailles de fichiers JPEG par sous-ensemble](../assets/figures/taille-fichiers.png)

## Observations clés

- Les **1 506 images** sont lisibles, au même format, en RGB et en 512 × 512 : le jeu est techniquement homogène.
- Les fichiers sont légers (13 à 74 KiB), ce qui rend les temps de lecture négligeables.
- **186 fichiers** sont impliqués dans des doublons binaires exacts. Ils font l'objet de l'analyse [Doublons](doublons.md).

!!! warning "Fuite de données"
    Les doublons doivent être regroupés ou retirés avant tout découpage entre apprentissage et test : une même image ne doit jamais se retrouver dans les deux.

## Livrables

| Élément | Emplacement |
|---|---|
| Notebook | `notebooks/01.01-kl-data-exploration.ipynb` |
| Inventaire et hash | `brain_mri.dataset.build_inventory`, `brain_mri.dataset.add_file_hash` |
