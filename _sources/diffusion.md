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

Soit $q(\mathbf{x}_0)$ la distribution des données réelles. On échantillonne (\mathbf{x}_0 \sim q(\mathbf{x}_0)$ et on définit un processus de diffusion avant $q(\mathbf{x}_t | \mathbf{x}_{t-1})$ qui ajoute un bruit gaussien à chaque pas de temps $t\in[\![1,T]\!]$, selon une mise à jour de la variance connue  $0 < \beta_1 < \beta_2 < ... < \beta_T < 1$ : 

$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})
$$

Chaque donnée $\boldsymbol x_t$ est ainsi tirée selon une distribution conditionnelle gaussienne  $\mathbf{\mu}_t = \sqrt{1 - \beta_t} \mathbf{x}_{t-1}$ et  $\sigma^2_t = \beta_t$,ce qui peut être réalisé en échantillonnant selon  $\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ et en posant  $\mathbf{x}_t = \sqrt{1 - \beta_t} \mathbf{x}_{t-1} +  \sqrt{\beta_t} \mathbf{\epsilon}$. 

En définissant les $\beta_t$ correctement, $\mathbf{x}_T$ est un bruit gaussien.

Si on connaissait la distribution conditionnelle  $p(\mathbf{x}_{t-1} | \mathbf{x}_t)$, alors on pourrait calculer le processus inverse en tirant $\mathbf{x}_T$ selon une distribution gaussienne isotrope, et en "débruitant" progressivement pour aboutir en $t=0$ à une réalisation $\mathbf{x}_0$$ de la distribution des données. Cependant, $p(\mathbf{x}_{t-1} | \mathbf{x}_t)$ n'est pas accessible et on utiliser un réseau de neurones $p_{\boldsymbol \theta} (\mathbf{x}_{t-1} | \mathbf{x}_t)$ pour approcher cette distribution conditionnelle, où $\boldsymbol \theta$ est l'ensemble des paramètres du réseau.

SI on suppose que le processus inverse est gaussien, alors on peut écrire 

$$ p_{\boldsymbol \theta} (\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_{t},t), \Sigma_\theta (\mathbf{x}_{t},t))$$
