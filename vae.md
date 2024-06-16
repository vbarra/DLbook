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
# Autoencodeurs variationnels
Les autoencodeurs variationnels sont, comme les GAN (abordés dans le cours suivant), des modèles génératifs, c'est-à-dire des modèles probabilistes $p$ pouvant être utilisés pour 
simuler (ou générer) des données réalistes $x\sim p(x,\boldsymbol\theta)$, aussi proches que possible de la vraie (mais inconnue) distribution des données $p(x)$, pour laquelle seul un échantillon de données est disponible.

Le paysage de ces modèles génératifs s'est beaucoup peuplé depuis 2014 ({numref}`landscape`).



```{figure} ./images/landscapegen.png
:name: landscape
Paysage des modèles génératifs (Source: [Song et al., CVPR 2023](https://cvpr2023.thecvf.com/virtual/2023/tutorial/18546))
```

Dans ce cours, nous aborderons uniquement les autoencodeurs variationnels (VAE) et les réseaux antagonistes générateurs (GAN).



## Inférence variationnelle

### Modèles à variables latentes

Un modèle à variables latentes met en relation un ensemble de variables observables $\boldsymbol x\in \mathcal X$ avec un ensemble de variables latentes  $\boldsymbol h\in \mathcal H$

$$\prob(\boldsymbol x,\boldsymbol h) = \prob(\boldsymbol x|\boldsymbol h)\prob(\boldsymbol h)$$

Si $\boldsymbol h$ sont des facteurs causaux pour  $\boldsymbol x$, alors échantillonner selon $\prob(\boldsymbol x|\boldsymbol h)$ permet de créer n modèle génératif de  $\mathcal H$ vers $\mathcal X$.

Pour l'inférence, étant donnée $\prob(\boldsymbol x,\boldsymbol h)$, il "suffit" de calculer 

$$ \prob(\boldsymbol h|\boldsymbol x) = \frac{\prob(\boldsymbol x|\boldsymbol h)\prob(\boldsymbol h)}{\prob(\boldsymbol x)}$$

Malheureusement, $\prob(\boldsymbol x)$ est inaccessible.

### Inférence variationnelle

L'inférence variationnelle transforme l'estimation de $\prob(\boldsymbol h|\boldsymbol x)$ en un problème d'optimisation.


## Autoencodeurs variationnels

