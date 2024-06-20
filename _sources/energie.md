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

# Modèles basés energie

Alors que la plupart des modèles précédents ont pour objectif initial la classification ou la régression, les modèles basés énergie sont plutôt orientés sur l'estimation de densité. Etant donné un ensemble $\mathcal E_a$ de données  $\boldsymbol x_i,i\in[\![1,n]\!]\in\mathcal X$, ces modèles recherchent à partir de $\mathcal E_a$ une distribution de probabilité $p(\boldsymbol x)$ sur $\mathcal X$ telle que $p(\boldsymbol x)$ est élevé si la donnée $\boldsymbol x$ "ressemble" à une donnée de $\mathcal E_a$.

L'idée des modèles basés énergie  est de transformer toute fonction qui prédit des valeurs positives en une distribution de probabilité en la divisant par son volume. En l'occurrence, les fonctions ciblées seront approchées par des réseaux de neurones.

Soit $f_{\boldsymbol\theta}$ un réseau de neurones, paramétré par $\boldsymbol\theta$. Etant donné $\boldsymbol x\in\mathcal X$, $f_{\boldsymbol\theta}(\boldsymbol x)\in\mathbb{R}$. On pose alors :

$$q_{\boldsymbol\theta}(\boldsymbol x) = \frac{e^{-f_{\boldsymbol\theta}(\boldsymbol x)}}{Z_{\boldsymbol\theta}}\textrm{ où } Z_{\boldsymbol\theta} = \left \{\begin{array}{cc} \int_{\boldsymbol x\in\mathcal X} e^{-f_{\boldsymbol\theta}(\boldsymbol x)} dx & \textrm{si } \boldsymbol x \textrm{ est discret}\\ \\displaystyle\sum_{\boldsymbol x\in\mathcal X} e^{-f_{\boldsymbol\theta}(\boldsymbol x)} dx & \textrm{si } \boldsymbol x \textrm{ sinon}\end{array}\right .$$


