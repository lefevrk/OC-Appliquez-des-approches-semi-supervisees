# Module `brain_mri.attention`

Cartes d'activation et Grad-CAM du ResNet18 pré-entraîné, pour voir **où le réseau regarde**. Le détail est décrit dans la page [Extraction des features](../pipeline/features.md#cartes-dactivation-et-dattention).

Ces cartes se calculent avec le ResNet18 **complet**, tête de classification comprise : l'extraction des features, elle, s'arrête à `layer3` et `layer4`.

!!! warning "Lecture des cartes"
    Le Grad-CAM porte sur la classe **ImageNet** prédite (objets et animaux du quotidien), qui n'a aucun sens médical. Une carte se lit comme « les zones qui comptent pour le réseau », pas comme un indice de cancer.

## Constantes

| Nom | Valeur | Rôle |
|---|---|---|
| `IMAGENET_CLASSES` | 1 000 noms de classes | Classes sur lesquelles le ResNet18 a été entraîné |

## Chargement

::: brain_mri.attention.load_classifier

## Cartes

::: brain_mri.attention.feature_maps

::: brain_mri.attention.grad_cam
