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