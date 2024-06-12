---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Réseaux antagonistes générateurs

Dans les réseaux antagonistes génératifs (GAN: Generative Adversarial Networks), la tâche d'apprentissage d'un modèle génératif est exprimée comme un jeu à somme nulle à deux joueurs entre deux réseaux, le générateur $G$ et le discriminateur $D$ ({numref}`gan1`).


```{figure} ./images/overview.png
:name: gan1
Architecture globale d'un GAN
```

Le générateur $G=G(.,\boldsymbol\theta): \mathcal Z\rightarrow \mathcal X$ met en correspondance un espace latent équipé d'une distribution $p(z)$ avec l'espace des données, induisant ainsi une distribution 

$$x\sim q(x,\boldsymbol\theta) \Leftrightarrow z\sim p(z), x=G(z,\boldsymbol\theta)$$