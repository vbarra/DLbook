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
# Modèles de diffusion

## Introduction

Un modèle de diffusion (de débruitage) transforme du bruit à partir d'une distribution simple en un échantillon de données. Lorsque les données sont des images, le modèle se compose de deux processus ({numref}`diffusion`): 
- un processus de diffusion vers l'avant $q$, choisi, qui ajoute progressivement du bruit gaussien à une image, jusqu'à aboutir à du bruit pur
- un processus de diffusion inverse de débruitage $p_{\boldsymbol\theta}$ , modélisé par un réseau de neurones, entraîné à débruiter progressivement une image à partir d'un bruit pur, jusqu'à obtenir une image réelle.

Les processus sont temporels, indexés par le temps $t\in[\![0,T]\!]$. A $t=0$, on  échantillonne une image réelle $\boldsymbol 𝐱_0$ de la distribution de données. Le processus $q$ échantillonne un bruit provenant d'une distribution gaussienne à chaque pas de temps $t$ ,  ajouté à l'image du pas de temps précédent. Si $T$ est suffisamment et que les processus d'ajout de bruit est correct, on obtient une distribution gaussienne isotrope à $t=T$.

```{prf:remark}
:class: dropdown
 Une distribution gaussienne isotrope $\mathcal N(\boldsymbol \mu,\boldsymbol \Sigma)$ est telle que $\boldsymbol \Sigma = \sigma^2\boldsymbol I$.
 ```


```{figure} ./images/diffusion.png
:name: diffusion
Illustration du modèle de diffusion (source : [{cite:p}`Ho20`](https://proceedings.neurips.cc/paper_files/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf))
```


## Modèle de diffusion

### Modèle
Soit $q(\mathbf{x}_0)$ la distribution des données réelles. On échantillonne $\mathbf{x}_0 \sim q(\mathbf{x}_0)$ et on définit un processus de diffusion avant $q(\mathbf{x}_t | \mathbf{x}_{t-1})$ qui ajoute un bruit gaussien à chaque pas de temps $t\in[\![1,T]\!]$, selon une mise à jour de la variance connue  $0 < \beta_1 < \beta_2 < ... < \beta_T < 1$ : 

$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})
$$

Chaque donnée $\boldsymbol x_t$ est ainsi tirée selon une distribution conditionnelle gaussienne  $\mathbf{\mu}_t = \sqrt{1 - \beta_t} \mathbf{x}_{t-1}$ et  $\sigma^2_t = \beta_t$,ce qui peut être réalisé en échantillonnant selon  $\mathbf{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ et en posant  $\mathbf{x}_t = \sqrt{1 - \beta_t} \mathbf{x}_{t-1} +  \sqrt{\beta_t} \mathbf{\varepsilon}$. 

En définissant les $\beta_t$ correctement, $\mathbf{x}_T$ est un bruit gaussien.

Si on connaissait la distribution conditionnelle  $p(\mathbf{x}_{t-1} | \mathbf{x}_t)$, alors on pourrait calculer le processus inverse en tirant $\mathbf{x}_T$ selon une distribution gaussienne isotrope, et en "débruitant" progressivement pour aboutir en $t=0$ à une réalisation $\mathbf{x}_0$ de la distribution des données. Cependant, $p(\mathbf{x}_{t-1} | \mathbf{x}_t)$ n'est pas accessible et on utilise un réseau de neurones $p_{\boldsymbol \theta} (\mathbf{x}_{t-1} | \mathbf{x}_t)$ pour approcher cette distribution conditionnelle, où $\boldsymbol \theta$ est l'ensemble des paramètres du réseau.

Si on suppose que le processus inverse est gaussien, alors on peut écrire 

$$ p_{\boldsymbol \theta} (\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_{t},t), \Sigma_\theta (\mathbf{x}_{t},t))$$

Le réseau doit donc apprendre la moyenne et la variance, qui dependent du temps. Dans l'implémentation initiale {cite:p}`Ho20`, les auteurs relaxent la contrainte de la variance ($\Sigma_\theta ( \mathbf{x}_t, t) = \sigma^2_t \mathbf{I}$), et apprennent uniquement la moyenne. Dans les implémentations suivantes (par exemple [celle-ci](https://openreview.net/pdf?id=-NEXDKk8gZ)), la contrainte a été prise en compte.

Dans la suite, on suppose seulement apprendre la moyenne.

### Apprentissage


La combinaison de $q$ et $p_{\boldsymbol\theta}$ peut-être vue comme un autoencodeur variationnel. Ainsi, ELBO peut être utilisée pour minimiser la log-vraisemblance négative par rapport à $\mathbf{x}_0$ . ELBO est la somme de fonctions de pertes calculées à chaque pas de temps 

$\ell  = \displaystyle\sum_{t=0}^T \ell_t$. 

Par construction de $q$ et $p_{\boldsymbol\theta}$, les $\ell_t,t\in[\![1,T]\!]$ sont les divergences de Kullback-Leibler entre deux distributions gaussiennes, ce qui peut être écrit comme une perte $\ell_2$ calculée sur les moyennes de ces gaussiennes.

Par construction de $q$, puisque la somme de gaussiennes est égalemment gaussienne, on peut échantillonner $\mathbf{x}_t$ pour tout $t$ conditionnellement à $\mathbf{x}_0$ : ainsi 

$$q(\mathbf{x}_t | \mathbf{x}_0) = \cal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1- \bar{\alpha}_t) \mathbf{I})$$

avec  $\alpha_t := 1 - \beta_t$ et  $\bar{\alpha_t} := \displaystyle\prod_{s=1}^{t} \alpha_s$. 

Les $\bar{\alpha}_t$ sont des fonctions des  $\beta_t$, permettant de mettre à jour la variance. Ces derniers étant connus, les $\bar{\alpha}_t$ le sont aussi et peuvent être précalculés. 
Ainsi, pendant l'entraînement, on tire aléatoirement $t$ et on optimise $\ell_t$

Il est également possible de reparamétriser la moyenne pour que le réseau de neurones apprenne le bruit ajouté (via un réseau $\mathbf{\varepsilon}_{\boldsymbol \theta}(\mathbf{x}_t, t)$ pour un niveau de bruit $t$ dans les divergences de Kullback-Leibler définissant les $\ell_t$. Le réseau $p_{\boldsymbol \theta}$ prédit donc le bruit plutôt que la moyenne, qui peut ensuite être calculée par 

$$ \mathbf{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left(  \mathbf{x}_t - \frac{\beta_t}{\sqrt{1- \bar{\alpha}_t}} \mathbf{\varepsilon}_\theta(\mathbf{x}_t, t) \right)$$

La fonction objectif finale $\ell_t$, à $t$ choisi aléatoirement et étant donné $\mathbf{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ : 

$$ \| \mathbf{\varepsilon} - \mathbf{\varepsilon}_\theta(\mathbf{x}_t, t) \|^2 = \| \mathbf{\varepsilon} - \mathbf{\varepsilon}_\theta( \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{(1- \bar{\alpha}_t)  } \mathbf{\varepsilon}, t) \|^2.$$

avec $\mathbf{x}_0$ la donnée initiale. $\mathbf{\varepsilon}$ est le bruit échantillonné au temps $t$ et $\mathbf{\varepsilon}_\theta (\mathbf{x}_t, t)$ est le réseau de neurones, équipé d'une fonction de perte quadratique entre le bruit réel et le bruit gaussien prédit.


```{prf:algorithm} Algorithme d'apprentissage d'un modèle de diffusion 
1. Tant que (non convergence)
    1. $\boldsymbol x_0\sim q(\boldsymbol x_0)$
    2. $t\sim Uniform(1,T)$
    3. $\varepsilon\sim\mathcal N(\boldsymbol 0,\boldsymbol I)$
    4. Calculer $\nabla_{\boldsymbol\theta}\|\mathbf\varepsilon - \mathbf{\varepsilon}_\theta(\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{(1- \bar{\alpha}_t)  } \mathbf{\varepsilon}, t) \|^2$
    5. Mettre à jour $\boldsymbol\theta$ par un pas de descente de gradient
```
Plutôt qu'un seul exemple $x_0$, le réseau est classiquement entraîné sur un batch.