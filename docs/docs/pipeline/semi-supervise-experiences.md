# Expérimentations et comparaison quantitative

Notebook : `notebooks/03.03-kl-semi-supervised-experiments.ipynb`

La page [Méthode semi-supervisée](semi-supervise.md) compare un CNN entraîné sur les labels forts seuls (A) à un CNN entraîné sur les labels faibles puis sur les labels forts (C), avec une seule graine et des réglages fixés d'avance. Cette page justifie ces réglages par deux **expérimentations** et répète les entraînements sur **plusieurs graines** pour voir si l'écart entre C et A est constant.

## Les variantes

| Variante | Couches entraînées | Époques sur les labels faibles | Graines |
|---|---|---|---|
| **référence** (celle de la méthode semi-supervisée) | `layer4` et couche de classification | 10 | 5 (42 à 46) |
| **tête seule** | couche de classification seule | 10 | 3 (42 à 44) |
| **labels faibles courts** | `layer4` et couche de classification | 3 | 3 (42 à 44) |

*Tête seule* teste si dégeler `layer4` est utile avec si peu d'images annotées. *Labels faibles courts* teste si une phase plus courte limite la mémorisation des labels faibles. Tous les autres réglages (Adam à 1e-4, lots de 16, arrêt anticipé de patience 5, augmentation légère) ne changent pas.

**Choix de la variante.** La variante retenue est celle dont le modèle C a le meilleur rappel de validation moyen, puis le meilleur F1 macro de validation. Les mesures de test sont rapportées pour toutes les variantes sans servir à choisir.

Chaque entraînement (A, B ou C, pour chaque variante et chaque graine) est un run MLflow. Les modèles eux-mêmes ne sont pas enregistrés.

## Résultats par variante

Moyenne ± écart-type sur les graines de la variante.

**Validation (14 images)**

| Variante | Modèle | Rappel `cancer` | F1 macro |
|---|---|---:|---:|
| référence | A | 1,000 ± 0,000 | 0,884 ± 0,041 |
| | B | 0,543 ± 0,064 | 0,692 ± 0,037 |
| | C | 0,829 ± 0,064 | 0,814 ± 0,064 |
| tête seule | A | 0,952 ± 0,082 | 0,743 ± 0,137 |
| | B | 0,619 ± 0,082 | 0,734 ± 0,044 |
| | C | 0,714 ± 0,000 | 0,785 ± 0,000 |
| labels faibles courts | A | 1,000 ± 0,000 | 0,879 ± 0,043 |
| | B | 0,667 ± 0,082 | 0,759 ± 0,044 |
| | C | 0,857 ± 0,000 | 0,809 ± 0,042 |

**Test (15 images)**

| Variante | Modèle | Rappel `cancer` | F1 macro |
|---|---|---:|---:|
| référence | A | 0,825 ± 0,112 | 0,906 ± 0,060 |
| | B | 0,425 ± 0,068 | 0,674 ± 0,044 |
| | C | 0,875 ± 0,000 | 0,933 ± 0,000 |
| tête seule | A | 0,875 ± 0,217 | 0,786 ± 0,081 |
| | B | 0,250 ± 0,000 | 0,550 ± 0,000 |
| | C | 0,375 ± 0,217 | 0,632 ± 0,142 |
| labels faibles courts | A | 0,792 ± 0,072 | 0,888 ± 0,039 |
| | B | 0,333 ± 0,072 | 0,611 ± 0,053 |
| | C | 0,833 ± 0,072 | 0,911 ± 0,039 |

![Rappel de cancer selon la variante et la graine](../assets/figures/experiences-rappel.png)

- **Expérience 1 : dégeler `layer4` est utile.** Avec la seule couche de classification, A perd en précision (0,82 contre 1,00 sur le test : des fausses alertes) et en F1 macro de validation (0,74 contre 0,88), et C devient nettement moins bon que A (rappel de test de 0,38 contre 0,88). Les labels faibles tirent probablement une couche linéaire vers leur partition sans que les 68 labels forts la corrigent ; cette explication est une hypothèse, pas une mesure.
- **Expérience 2 : raccourcir la phase faible change peu.** B s'améliore sur la validation (rappel de 0,67 contre 0,54), mais C ne change presque pas (rappel de validation de 0,86 contre 0,83, de test de 0,83 contre 0,88).
- **Variante retenue : labels faibles courts**, avec un rappel de validation de C de 0,857 contre 0,829. Cet écart de 0,03 est inférieur à celui d'une image de validation (0,14 sur 7 cancers) : le choix n'est pas démontré.
- **A est meilleur que C sur la validation** dans toutes les variantes, validation qui sert à l'arrêt anticipé de A et de C.

## A contre C, graine par graine

Le tableau compare, sur chaque graine, le rappel de `cancer` de C et celui de A sur les mêmes 15 images de test.

| Variante | Graine | Rappel de A | Rappel de C | C moins A |
|---|---:|---:|---:|---:|
| référence | 42 | 0,750 | 0,875 | +0,125 |
| | 43 | 0,750 | 0,875 | +0,125 |
| | 44 | 0,875 | 0,875 | 0,000 |
| | 45 | 1,000 | 0,875 | −0,125 |
| | 46 | 0,750 | 0,875 | +0,125 |
| tête seule | 42 | 0,625 | 0,625 | 0,000 |
| | 43 | 1,000 | 0,250 | −0,750 |
| | 44 | 1,000 | 0,250 | −0,750 |
| labels faibles courts | 42 | 0,750 | 0,875 | +0,125 |
| | 43 | 0,750 | 0,750 | 0,000 |
| | 44 | 0,875 | 0,875 | 0,000 |

| Variante | Graines | C meilleur | Égal | C moins bon | Rappel, C moins A (moyenne) | F1 macro, C moins A (moyenne) |
|---|---:|---:|---:|---:|---:|---:|
| référence | 5 | 3 | 1 | 1 | +0,050 | +0,027 |
| tête seule | 3 | 0 | 1 | 2 | −0,500 | −0,154 |
| labels faibles courts | 3 | 1 | 2 | 0 | +0,042 | +0,022 |

- **Référence.** C fait mieux que A sur 3 graines sur 5, autant sur une et moins bien sur une. Son rappel de test est le même (0,875) à chaque graine, alors que celui de A varie de 0,75 à 1,00 : l'avantage moyen de 0,05 vient des trois graines où A est à 0,75, et disparaît quand A est à 0,875 ou plus.
- **Labels faibles courts (variante retenue).** C fait mieux que A sur 1 graine sur 3 et autant sur les 2 autres.
- **Tête seule.** C ne fait jamais mieux que A, et fait nettement moins bien sur 2 graines sur 3.

## Observations

- **Dégeler `layer4` est justifié.** Avec la seule couche de classification, A fait des fausses alertes et C devient nettement moins bon que A.
- **Raccourcir la phase sur les labels faibles change peu** : B s'améliore sur la validation, C presque pas.
- **A contre C.** Sur la référence, C fait mieux que A ou autant sur 4 graines sur 5, avec un rappel de test constant (0,875), dans le sens attendu d'une approche semi-supervisée. L'écart est d'une image : avec 15 images de test, on ne peut pas conclure.

!!! warning "Limites"
    - **Test de 15 images** : un cancer raté représente 12,5 points de rappel. Un écart d'une image entre deux modèles n'est pas significatif : on ne peut pas conclure avec si peu de données.
    - **Peu de graines** : 5 pour la référence, 3 pour les autres variantes.
    - **Choix sur 14 images de validation** : l'écart qui désigne la variante retenue est inférieur à une image.
    - **Peu de variantes** : deux changements seulement (couches entraînées, durée de la phase faible) ; taux d'apprentissage, taille des lots, patience et augmentation ne changent pas.
    - **Normalisation par lot** : ses statistiques continuent de s'adapter aux IRM, même quand `layer4` est gelée.
    - **Labels faibles de qualité limitée** : leur effet est le même dans toutes les variantes.
