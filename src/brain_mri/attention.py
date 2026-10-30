import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

# Les 1 000 classes d'ImageNet, sur lesquelles le ResNet18 a été entraîné.
IMAGENET_CLASSES = ResNet18_Weights.DEFAULT.meta["categories"]


def load_classifier(device: torch.device) -> nn.Module:
    """Charge le ResNet18 complet (tête de classification comprise), gelé.

    Les poids sont téléchargés au premier appel.

    Args:
        device: Appareil de calcul.

    Returns:
        Le modèle en mode évaluation, dont aucun paramètre n'est entraînable.
    """
    model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model.to(device)


def _get_layer(model: nn.Module, layer: str) -> nn.Module:
    """Retourne la couche nommée, ou lève une erreur si elle n'existe pas."""
    module = getattr(model, layer, None)
    if not isinstance(module, nn.Module):
        raise ValueError(f"Le modèle n'a pas de couche `{layer}`.")
    return module


@torch.inference_mode()
def feature_maps(
    model: nn.Module, image: torch.Tensor, layer: str
) -> torch.Tensor:
    """Retourne les cartes de caractéristiques d'une couche, pour une image.

    Args:
        model: Modèle dont la couche est observée.
        image: Image prétraitée, de forme (canaux, hauteur, largeur).
        layer: Nom de la couche (par exemple `layer4`).

    Returns:
        Les cartes (canaux de la couche, hauteur, largeur), sur le CPU.

    Raises:
        ValueError: Si le modèle n'a pas de couche de ce nom.
    """
    module = _get_layer(model, layer)
    device = next(model.parameters()).device
    captured: dict[str, torch.Tensor] = {}
    handle = module.register_forward_hook(
        lambda module, inputs, output: captured.update(maps=output)
    )
    try:
        model(image[None].to(device))
    finally:
        handle.remove()
    return captured["maps"][0].cpu()


def grad_cam(
    model: nn.Module, batch: torch.Tensor, layer: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcule la carte Grad-CAM de la classe prédite, à la taille de l'image.

    Les cartes de caractéristiques de la couche sont pondérées par le gradient
    du score de la classe prédite, seules les contributions positives sont
    gardées, puis la carte est agrandie par interpolation bilinéaire et
    normalisée entre 0 et 1.

    Args:
        model: Modèle de classification, dont les paramètres sont gelés.
        batch: Images prétraitées, de forme (images, canaux, hauteur, largeur).
        layer: Nom de la couche observée (par exemple `layer4`).

    Returns:
        Les cartes (images, hauteur, largeur) normalisées par leur maximum, la
        classe prédite et sa probabilité pour chaque image.

    Raises:
        ValueError: Si le modèle n'a pas de couche de ce nom.
    """
    module = _get_layer(model, layer)
    device = next(model.parameters()).device
    captured: dict[str, torch.Tensor] = {}
    handle = module.register_forward_hook(
        lambda module, inputs, output: captured.update(maps=output)
    )
    try:
        # Les paramètres sont gelés : c'est l'entrée qui porte le gradient.
        batch = batch.to(device).clone().requires_grad_(True)
        logits = model(batch)
    finally:
        handle.remove()
    predicted = logits.argmax(dim=1)
    score = logits.gather(1, predicted[:, None]).sum()
    (gradient,) = torch.autograd.grad(score, captured["maps"])
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * captured["maps"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(
        cam, size=batch.shape[-2:], mode="bilinear", align_corners=False
    )[:, 0]
    cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    probabilities = torch.softmax(logits, dim=1).gather(1, predicted[:, None])
    return (
        cam.detach().cpu().numpy(),
        predicted.cpu().numpy(),
        probabilities[:, 0].detach().cpu().numpy(),
    )
