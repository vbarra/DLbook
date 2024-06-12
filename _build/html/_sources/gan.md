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

Le discriminateur $D=D(.,\boldsymbol\phi): \mathcal X\rightarrow [0,1]$ est un classifieur entraîné pour distinguer les vrais exemples $x\sim p(x)$ des exemples générés par $G$ $x\sim q(x,\boldsymbol\theta)$

Pour un $G$ fixé, $D$ peut-être entraîné en construisant un ensemble d'apprentissage

$$\mathcal E_a = \{(x_i,1),i\in[\![1,N]\!],(G(z_i,\boldsymbol\theta),1),i\in[\![1,N]\!]\}$$

où $x_i\sim  p(x)$ et $z_i\sim  p(z)$, et en minimisant l'entropie croisée binaire

$$\begin{align}
\mathcal L(\boldsymbol\phi)&=& -\frac{1}{2N}\displaystyle\sum_{i=1}^N \left [log D(x_i,\boldsymbol \phi)+ log (1-D(G(z_i,\boldsymbol\theta),\boldsymbol\phi))\right ]\\
&\approx& -\mathbb{E}_{x\sim p(x)}[log D(x_i,\boldsymbol \phi)] - \mathbb{E}_{z\sim p(z)}log (1-D(G(z_i,\boldsymbol\theta),\boldsymbol\phi))
\end{align}
$$