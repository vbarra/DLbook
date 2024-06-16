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
simuler (ou générer) des données réalistes $\boldsymbol x\sim p(\boldsymbol x,\boldsymbol\theta)$, aussi proches que possible de la vraie (mais inconnue) distribution des données $p(\boldsymbol x)$, pour laquelle seul un échantillon de données est disponible.

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

Si $\boldsymbol h$ sont des facteurs causaux pour  $\boldsymbol x$, alors échantillonner selon $p(\boldsymbol x|\boldsymbol h)$ permet de créer un modèle génératif de  $\mathcal H$ vers $\mathcal X$.

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
&=& \mathbb{E}_{ q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)}  \left [\color{blue}{log(p(\boldsymbol{x}|\boldsymbol h))}\right ]-\color{red}{KL( q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)|| p(\boldsymbol h))}
\end{align}$$

Dans l'équation prédécente, on a éliminé $log(p(\boldsymbol x))$ qui ne dépend pas de $\boldsymbol\phi$.

En maximisant la fonction $ELBO(\boldsymbol x,\boldsymbol\phi)$ :
- <span style="color:blue"> le premier terme encourage les distributions à converger dans les configurations des variables latentes $\boldsymbol h$ expliquant les données observées </span>
- <span style="color:red"> le second terme force les distribution à être proches du prior.</span>

Finalement, étant donné un échantillon  $E_a = \{\boldsymbol x_i,i\in[\![1,N]\!]\}$, la fonction objectif finale est 

$$\displaystyle\sum_{\boldsymbol x_i\in E_a}ELBO(\boldsymbol x_i,\boldsymbol \phi)$$

Pour la maximiser, on peut utiliser une montée de gradient, maix $\nabla_{\boldsymbol\phi}ELBO(\boldsymbol x_i,\boldsymbol \phi)$ est en général difficile à calculer.


## Autoencodeurs variationnels

Un autoencodeur variationnel (VAE) est un modèle profond à variables latentes ({numref}`vae0`) tel que :
- $p(\boldsymbol h)$ est prescrit à l'avance
- la vraisemblance $p_{\boldsymbol\theta}(\boldsymbol x|\boldsymbol h)$ est un décodeur (réseau génératif) $D_{\boldsymbol\theta}$ tel que $\boldsymbol\Phi = D_{\boldsymbol\theta}(\boldsymbol h)$ où $\boldsymbol \Phi$  sont les paramètres de la distribution des données. Par exemple 

$$\mu,\sigma  = D_{\boldsymbol\theta}(\boldsymbol h), \quad p_{\boldsymbol\theta}(\boldsymbol x|\boldsymbol h) = \mathcal N(\boldsymbol x,\mu,\sigma^2 \boldsymbol I)$$
- la distribution approchée $q_{\boldsymbol\phi}(\boldsymbol h|\boldsymbol x)$ est paramétrée par un encodeur (réseau d'inférence) $E_{\boldsymbol\phi}$  tel que $\boldsymbol\nu = E_{\boldsymbol\phi}(\boldsymbol x)$ sont les paramètres de la distribution approchée. Par exemple : 

$$ \mu,\sigma  = \boldsymbol\nu = E_{\boldsymbol\phi}(\boldsymbol x)\quad q_{\boldsymbol\phi}(\boldsymbol h|\boldsymbol x) = \mathcal N(\boldsymbol h,\mu,\sigma^2 \boldsymbol I)$$



```{figure} ./images/vae0.png
:name: vae0
Architecture générale d'un autoencodeur variationnel
```

Comme précédemment, on peut utiliser l'inférence variationnelle pour optimiser de manière jointe $\boldsymbol\theta$ et $\boldsymbol\phi$ : 

$$
\begin{align}
\boldsymbol\theta^*,\boldsymbol\phi^* &=& Arg\; max_{\boldsymbol\theta,\boldsymbol\phi}\; ELBO(\boldsymbol x,\boldsymbol\theta,\boldsymbol\phi)\\
&=&Arg\; max_{\boldsymbol\theta,\boldsymbol\phi}\left [\mathbb{E}_{q_{\boldsymbol\phi}(\boldsymbol h|\boldsymbol x)}\left (log(p_{\boldsymbol\theta}(\boldsymbol x|\boldsymbol h))\right )-KL(q_{\boldsymbol\phi}(\boldsymbol h|\boldsymbol x)||p(\boldsymbol h))\right ]
\end{align}
$$

Etant donné $D_{\boldsymbol\theta}$, on veut ajuster les variables latentes, en optimissant $\boldsymbol\phi$, de sorte à ce qu'elles expliquent les données observées, tout en restant près de données générées par $p(\boldsymbol h)$. De même, étant donné $E_{\boldsymbol\phi}$, on veut ajuster les variables observées, en optimissant $\boldsymbol\theta$, de telle sorte qu'elles soient le plus possible expliquées par les variables latentes.

L'optimisation peut se faire par montée de gradients : 
- $\nabla_{\boldsymbol\theta}ELBO(\boldsymbol x,\boldsymbol\theta,\boldsymbol\phi) = \mathbb{E}_{q_{\boldsymbol\phi}(\boldsymbol h|\boldsymbol x)}\left [\nabla_{\boldsymbol\theta}(log p_{\boldsymbol\theta}(\boldsymbol x|\boldsymbol h))\right ]$ peut être estimé (par exemple par méthode de Monte Carlo)
- $\nabla_{\boldsymbol\phi}ELBO(\boldsymbol x,\boldsymbol\theta,\boldsymbol\phi)$ est plus difficile à estimer (on ne peut rétropropager le gradient à travers $\boldsymbol h$ pour calculer $\nabla_{\boldsymbol\phi}$)

La solution à ce problème est appelée *astuce de reparamétrisation* (reparameterization trick) : on exprime $\boldsymbol h$ à l'aide d'une transformation différentiable et inversible d'une autre variable aléatoire $\varepsilon$, étant donnés $\boldsymbol x$ et ${\boldsymbol\phi$

$$\boldsymbol h = \mu(\boldsymbol x,\phi) + \sigma(\boldsymbol x,\phi)\odot \varepsilon,\quad \varepsilon\sim \mathcal{N}(0,I)$$ 

de telle sorte que la distribution de $\varepsilon$ est indépendante de  $\boldsymbol x$ et ${\boldsymbol\phi$.


