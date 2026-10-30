# Module `brain_mri.modeling.unsupervised`

Mesure de l'accord entre clusters et labels, attribution de labels faibles avec K-Means, et évaluation de ces labels. Le détail du flux est décrit dans la page [Analyse non supervisée](../pipeline/analyse-non-supervisee.md).

## Constantes

| Nom | Valeur | Rôle |
|---|---|---|
| `REFERENCE_LAYER` | `"layer4"` | Couche dont les features servent aux labels faibles |
| `N_CLUSTERS` | `2` | Nombre de clusters de K-Means, comme de classes |
| `NOISE_LABEL` | `-1` | Cluster du bruit (DBSCAN) |
| `CLASS_CODES` | `cancer` : 1, `normal` : 0 | Codage des labels forts |
| `CONFIDENCE_LEVEL` | `0.95` | Niveau des intervalles de confiance |

## Labels forts et accord avec les clusters

::: brain_mri.modeling.unsupervised.encode_labels

::: brain_mri.modeling.unsupervised.ari_without_noise

## Labels faibles

::: brain_mri.modeling.unsupervised.majority_mapping

::: brain_mri.modeling.unsupervised.to_weak_labels

::: brain_mri.modeling.unsupervised.kmeans_confidence

::: brain_mri.modeling.unsupervised.build_weak_labels

## Évaluation

::: brain_mri.modeling.unsupervised.wilson_interval

::: brain_mri.modeling.unsupervised.WeakLabelEvaluation

::: brain_mri.modeling.unsupervised.evaluate_weak_labels

## Ligne de commande

::: brain_mri.modeling.unsupervised.main
