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

## Principe
Dans les réseaux antagonistes génératifs (GAN: Generative Adversarial Networks), la tâche d'apprentissage d'un modèle génératif est exprimée comme un jeu à somme nulle à deux joueurs entre deux réseaux, le générateur $G$ et le discriminateur $D$ ({numref}`gan1`).


```{figure} ./images/overview.png
:name: gan1
Architecture globale d'un GAN
```

Le générateur $G=G(.,\boldsymbol\theta): \mathcal Z\rightarrow \mathcal X$ met en correspondance un espace latent équipé d'une distribution $p(z)$ avec l'espace des données, induisant ainsi une distribution 

$$x\sim q(x,\boldsymbol\theta) \Leftrightarrow z\sim p(z), x=G(z,\boldsymbol\theta)$$

Le discriminateur $D=D(.,\boldsymbol\phi): \mathcal X\rightarrow [0,1]$ est un classifieur entraîné pour distinguer les vrais exemples $x\sim p(x)$ des exemples générés par $G$ $x\sim q(x,\boldsymbol\theta)$

Pour un $G$ fixé, $D$ peut-être entraîné en construisant un ensemble d'apprentissage

$$\mathcal E_a = \{(x_i,1),i\in[\![1,N]\!],(G(z_i,\boldsymbol\theta),0),i\in[\![1,N]\!]\}$$

où $x_i\sim  p(x)$ et $z_i\sim  p(z)$, et en minimisant l'entropie croisée binaire

$$\begin{align}
\mathcal L(\boldsymbol\phi)&=& -\frac{1}{2N}\displaystyle\sum_{i=1}^N \left [log D(x_i,\boldsymbol \phi)+ log (1-D(G(z_i,\boldsymbol\theta),\boldsymbol\phi))\right ]\\
&\approx& -\mathbb{E}_{x\sim p(x)}[log D(x_i,\boldsymbol \phi)] - \mathbb{E}_{z\sim p(z)}log (1-D(G(z_i,\boldsymbol\theta),\boldsymbol\phi))
\end{align}
$$

Cependant, la situation est légèrement plus complexe puisque $G$ doit également être entraîné pour tromper $D$. Heureusement, cela revient à maximiser la perte de $D$.

Soit 

$$f(\boldsymbol\theta,\boldsymbol\phi) = \mathbb{E}_{x\sim p(x)}[log D(x_i,\boldsymbol \phi)] + \mathbb{E}_{z\sim p(z)}log (1-D(G(z_i,\boldsymbol\theta),\boldsymbol\phi))$$

Alors : 
- pour $G$ fixé $f(\boldsymbol\theta,\boldsymbol\phi)$ est élevée si $D$ discrimine les vrais exemples des faux
- si $\hat D$ est le meilleur classifieur étnt donné $G$, et si $f(\boldsymbol\theta,\boldsymbol\phi)$ est élevée, alors $G$ n'arrive pas à reproduire la distribution des données $p(x)$
- Inversement, $G$ est un bon modèle génératif, capable de reproduire $p(x)$, si $f(\boldsymbol\theta,\boldsymbol\phi)$ est faible lorsque $\hat D$ est le meilleur classifieur.

Tout ceci implique que l'on recherche $G=G(.,\boldsymbol\theta^*)$ tel que 

$$\boldsymbol\theta^* = Arg min_{\boldsymbol\theta}max_{\boldsymbol\phi}f(\boldsymbol\theta,\boldsymbol\phi)$$

En pratique, ce problème d'optimisation minimax est approché par une descente de gradient par batch alternée : 

1. $\boldsymbol\phi = \boldsymbol\phi+\eta\nabla_{\boldsymbol\phi}f(\boldsymbol\theta,\boldsymbol\phi)$ ({numref}`gan2`)
```{figure} ./images/overview2.png
:name: gan2
Optimisation des paramètres de $D$, à $G$ fixé
```

2. $\boldsymbol\theta = \boldsymbol\theta-\eta\nabla_{\boldsymbol\theta}f(\boldsymbol\theta,\boldsymbol\phi)$ ({numref}`gan3`)

```{figure} ./images/overview3.png
:name: gan3
Optimisation des paramètres de $G$, à $D$ fixé
```

Goodfellow propose dans {cite:p}`Goodfellow14` une illustration du processus d'apprentissage d'un GAN ({numref}`gan4`). Dans cette figure, on illustre de gauche à droite un modèle à l'état initial, l'évolution après la mise à jour de $D$, puis de $G$, et enfin l'état à convergence. La courbe en points noir est la distribution des données $p(x)$, la courbe verte la distribution $p(z)$ de $G$, la courbe bleue pointillée la sortie de $D$. Les $z\in\mathcal Z$ sont tirés uniformément (partie inférieure des graphiques)


```{figure} ./images/illustrationGan1D.png
:name: gan4
Processus d'apprentissage d'un GAN.
```

## Analyse
A $G$ fixé (donc $\boldsymbol\theta$ fixé), le classifieur $D$ de paramètres optimaux $\boldsymbol\phi_{\boldsymbol\theta}^*$  est optimal si et seulement si 

$$\forall x, D(x,\boldsymbol\phi_{\boldsymbol\theta}^*) = \frac{p(x)}{p(x)+q(x\boldsymbol\theta)}$$

En notant $KL(p||q)$ la divergence de Kullback-Leibler entre $p$ et $q$ et $JSD$ la divergence de Jensen-Shannon, on a ainsi : 

$$
\begin{aligned}
& \min _\boldsymbol\theta \max _\boldsymbol\phi f(\boldsymbol\theta,\boldsymbol\phi)=\min _\boldsymbol\theta f(\boldsymbol\theta,\boldsymbol\phi)_{\boldsymbol\theta^*}) \\
& =\min _\boldsymbol\theta \mathbb{E}_{x \sim p(x)}\left[\log \frac{p(x)}{q(x ; \boldsymbol\theta)+p(x)}\right]+\mathbb{E}_{x \sim q(x ; \theta)}\left[\log \frac{q(x ; \boldsymbol\theta)}{q(x ; \boldsymbol\theta)+p(x)}\right] \\
& =\min _\boldsymbol\theta \mathrm{KL}\left(p(x) \| \frac{p(x)+q(x ; \boldsymbol\theta)}{2}\right) \\
& \quad+\mathrm{KL}\left(q(x ; \boldsymbol\theta) \| \frac{p(x)+q(x ; \boldsymbol\theta)}{2}\right)-\log 4 \\
& =\min _\boldsymbol\theta 2 \mathrm{JSD}(p(x) \| q(x ; \boldsymbol\theta))-\log 4
\end{aligned}
$$

```{prf:remark}
:class: dropdown
 La divergence de Kullback-Leibler est une mesure de dissimilarité entre deux distributions de probabilités. Pour deux distributions $p$ et $q$, elle est définie par 
 
 $$KL(p\|q) = \sum_{x \in X} p(x) \log \frac{p(x)}{q(x)}$$

La divergence de Jensen-Shannon est une autre méthode de mesure de la similarité entre deux distributions de probabilité, définie par 

$$JSD(p,q)=\frac12 (KL(p\|m) + KL(q\|m))\quad\textrm{avec }m = \frac12(p+q)$$

 ```
