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

$$p(\boldsymbol x,\boldsymbol h) = p(\boldsymbol x|\boldsymbol h)p(\boldsymbol h)$$

Si $\boldsymbol h$ sont des facteurs causaux pour  $\boldsymbol x$, alors échantillonner selon $p(\boldsymbol x|\boldsymbol h)$ permet de créer n modèle génératif de  $\mathcal H$ vers $\mathcal X$.

Pour l'inférence, étant donnée $p(\boldsymbol x,\boldsymbol h)$, il "suffit" de calculer 

$$ p(\boldsymbol h|\boldsymbol x) = \frac{p(\boldsymbol x|\boldsymbol h)p(\boldsymbol h)}{p(\boldsymbol x)}$$

Malheureusement, $p(\boldsymbol x)$ est inaccessible.

### Inférence variationnelle

L'inférence variationnelle transforme l'estimation de $p(\boldsymbol h|\boldsymbol x)$ en un problème d'optimisation.

On considère une famille de distributions $q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)$ approchant $p(\boldsymbol{h}|\boldsymbol x)$, où $\boldsymbol \phi$ sont les paramètres variationnels. Ces paramètres sont optimisés pour minimiser une distance entre $q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)$ et $p(\boldsymbol x,\boldsymbol h)$. Parmi toutes les distances possibles, on retient la divergence de Kullback-Leibler

 $$KL(p\|q) = \sum_{x \in X} p(x) \log \frac{p(x)}{q(x)}$$

et ainsi : 

$$\begin{align}
KL( q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)|| p(\boldsymbol h| \boldsymbol  x)) &=& \mathbb{E}_{ q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)} \left [log\frac{ q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)}{ p(\boldsymbol{h}|\boldsymbol x)} \right ]\\
&=& \mathbb{E}_{ q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)}  \left [ log(q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x))-log(p(\boldsymbol{x},\boldsymbol h))\right ] +log(p(\boldsymbol x))
\end{align}$$

Le dernier terme $log(p(\boldsymbol x)$ reste cependant toujours inaccessible.


## Autoencodeurs variationnels

