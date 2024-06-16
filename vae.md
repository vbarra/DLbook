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

Cependant, 

$$\begin{align}
\displaystyle\min_\boldsymbol\phi KL( q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)|| p(\boldsymbol h| \boldsymbol  x)) 
&=&\displaystyle\min_\phi log(p(\boldsymbol x))- \mathbb{E}_{ q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)}  \left [log(p(\boldsymbol{x},\boldsymbol h)) - log (q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x))\right ]\\
&=&\displaystyle\max_\phi \underbrace{\mathbb{E}_{ q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)}  \left [log(p(\boldsymbol{x},\boldsymbol h)) - log (q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x))\right ]}_{ELBO(\boldsymbol x,\phi)}
\end{align}$$

avec ELBO(Evidence Lower Bound Objective) définie par 

$$\begin{align}
ELBO(\boldsymbol x,\boldsymbol\phi)&=&\mathbb{E}_{ q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)}  \left [log(p(\boldsymbol{x},\boldsymbol h)) - log (q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x))\right ] \\
&=& \mathbb{E}_{ q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)}  \left [log(p(\boldsymbol{x}|\boldsymbol h))p(\boldsymbol h) - log (q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x))\right ]\\
&=& \mathbb{E}_{ q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)}  \left [\color{red}{log(p(\boldsymbol{x}|\boldsymbol h))}\right ]-<span style="color:red">KL( q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)|| p(\boldsymbol h))</span>
\end{align}$$

Maximiser la fonction $ELBO(\boldsymbol x,\boldsymbol\phi)$:
- <span style="color:blue">The first term encourages distributions to be centered on configurations of latent variables $\boldsymbol h$ explaining the observed data</span>
- <span style="color:red">The second term enforces distributions to be close to the prior.</span>



## Autoencodeurs variationnels

