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

# Réseaux récurrents

-----------------------------------------------------------------------------------------------

## Définition

Les réseaux de neurones récurrents (RNN, *Recurrent Neural Networks*)
sont des réseaux à propagation avant, permettant de prendre en compte le
temps. Comme dans les réseaux classiques, il n'existe pas de cycle, mais
les arcs ajoutés pour introduire la notion de temps (les arcs
récurrents) peuvent en revanche former des cycles, y compris de longueur
1 (connexion d'un neurone avec lui-même). À l'instant $t$, les neurones
possédant des arcs récurrents reçoivent en entrée la donnée courante
$\mathbf{x_t}$ et les valeurs des neurones cachés ${h_{t-1}}$ informant
sur l'état précédent du réseau. La sortie $\hat{y}_{t}$ est calculée
étant donné l'état $\mathbf{x_t}$ des neurones cachés à l'instant $t$.
La donnée $\mathbf{x_{t-1}}$ peut influencer $\hat{y}_{t}$ et la sortie
aux instants suivants, à l'aide des arcs récurrents.

Deux équations permettent de calculer les quantités nécessaires à
l'instant $t$ dans la phase de propagation avant d'un réseau récurrent
simple (comme celui de la {numref}`rnn1` gauche) : 

$$\begin{aligned}
h_t&=&\sigma\left ( \mathbf{W_{hx}^\top x_t} +  \mathbf{W_{hh}^\top x_{t-1}} + b_h\right)\\
\hat{y}_{t}&=&softmax\left({W_{yh}} h_t + b_y  \right)
\end{aligned}$$ 

où $\mathbf{W_{hx}}$ est la matrice des poids reliant
l'entrée à la couche cachée et $\mathbf{W_{hh}}$ celle des poids des
arcs récurrents. Les biais sont notés $b_h$ et $b_y$.\
La dynamique du réseau peut être décrite en dépliant ce réseau dans le
temps ({numref}`rnn1` droite). Le réseau devient donc un réseau profond, avec une couche par instant $t$ et un partage de poids au cours du temps. Ce dernier peut
donc être entraîné de manière classique par l'algorithme de
rétropropagation du gradient, indicé par le temps (*Backpropagation
through time*, BPTT algorithm).


```{figure} ./images/rnn1.png
:name: rnn1
Réseau récurrent et sa version dépliée dans le
temps.
```



Avec ces réseaux, il est possible de traiter des séquences de longueur
quelconque, la taille du modèle étant indépendante de cette longueur.
Plusieurs architectures peuvent être déclinées sur ce principe

```{figure} ./images/onetomany.png
:name: onetomany
Architectures "un vers plusieurs" , utilisées par exemple en génération de musique ou légendage d'images.
```

```{figure} ./images/manytoone.png
:name: onetomany
Architectures "un vers plusieurs" , utilisées par exemple en classification de sentiments
```
```{figure} ./images/manytomany1.png
:name: onetomany
Architectures "plusieurs vers plusieurs", pour la reconnaissance d'entité dans des textes ou annotation de vidéos.
```
```{figure} ./images/manytomany2.png
:name: onetomany
Architectures "plusieurs vers plusieurs", pour la traduction automatique.
```


## Entraînement des réseaux récurrents

L'apprentissage de dépendances long terme peut être difficile. Les
problèmes d'évanescence (*vanishing*) ou d'explosion du gradient peuvent
rapidement survenir, lors de la rétropropagation sur plusieurs pas de
temps.

Prenons un exemple simple pour comprendre : considérons un réseau à un
neurone d'entrée, un neurone récurrent caché et un neurone de sortie. On
donne au réseau une entrée à l'instant $t_0$ et on calcule l'erreur à
l'instant $t>t_0$, en supposant des entrées nulles entre $t_0$ et $t$.
Le lien entre les poids au cours du temps fait que le poids sur l'arc
récurrent ne change jamais. La contribution de l'entrée au temps $t_0$ à
la sortie au temps $t$ deviendra de plus en plus importante, ou se
rapprochera de zéro, de manière exponentielle à mesure que $t-t_0$
croît. Et la dérivée de l'erreur par rapport à l'entrée explosera ou
disparaîtra, selon que le poids de l'arc récurrent a une valeur absolue
plus grande ou plus petite que 1 et selon la fonction d'activation du
neurone caché (le problème du gradient évanescent est très présent avec
une sigmoïde et une activation ReLU force davantage l'explosion).

Plusieurs solutions ont été proposées (régularisation, retropropagation
tronquée, conception d'architecture et heuristiques) pour résoudre ces
problèmes.

## Quelques architectures

### LSTM

Les réseaux *Long Short-Term Memory* (LSTM) ont été introduits en 1997 {cite:p}`HO97` pour résoudre le problème de l'évanescence du gradient. Ce
modèle ressemble à un réseau récurrent classique à une couche cachée,
mais chaque neurone de la couche cachée est remplacé par une cellule de
mémoire.

Dans la suite, on note $\mathbf{x_t}$ l'entrée de la cellule à l'instant
$t$, ${h_{t-1}}$ la sortie de la couche cachée calculée au temps $t-1$.
Au lieu de calculer une sortie du type
$\sigma\left( \mathbf{W^\top x}+b\right)$, la cellule contient plusieurs
éléments distincts aux fonctions particulières. Les LSTM introduisent la
notion de portes, qui sont des unités d'activation de type sigmoïde qui
prennent comme arguments $\mathbf{x_t}$ et ${h_{t-1}}$ et viennent
pondérer des valeurs calculées dans la cellule. En particulier, si la
valeur d'une porte est nulle, alors le flot est coupé dans le graphe,
alors qu'il transite intégralement si la valeur de la porte est égale à
1.

On retrouve dans une cellule ({numref}`lstm1`) les éléments suivants :

-   *Neurone d'entrée* : ce neurone prend en entrée $\mathbf{x_t}$ et
    ${h_{t-1}}$ et calcule, à la manière d'un neurone classique, une
    sortie
    $g^{t} = \sigma\left(\mathbf{W_C^\top} \left[\mathbf{x_t},  {h_{t-1}}\right ] +b_C\right)$.

-   *Porte d'entrée* (ou de mise à jour) : la porte calcule
    $i^{t} = \sigma\left(\mathbf{W_i}^\top \left[\mathbf{x_t},  {h_{t-1}}\right ] +b_i\right)$
    et vient pondérer la valeur du neurone d'entrée pour décider de
    l'importance à lui donner au temps $t$.

-   Porte d'oubli : cette porte calcule
    $f^{t}=\sigma\left(\mathbf{W_f}^\top\left[\mathbf{x_t} , {h_{t-1}}\right ] +b_f\right)$
    et permet au réseau d'oublier son état interne.

-   *État interne* : le cœur de la cellule de mémoire est son état
    interne, noté $C^{t}$, composé d'un neurone récurrent à poids fixe
    unité, assurant que le gradient peut passer par cet arc de
    nombreuses fois sans disparaître ou exploser. La mise à jour de
    l'état interne est effectuée par une opération du type
    $C^{t} =g^{t}.i^{t} + C^{(t-1)}.f^{t}$.

-   *Porte de sortie* : la valeur $h_t$ produite par la cellule de
    mémoire est calculée comme le produit de $tanh(C^{t})$ par la valeur
    de la porte de sortie $o^{t}$. Cette porte sélectionne la part de
    $C^{t}$ à fournir en sortie et est calculée par
    $o^{t} = \sigma\left(\mathbf{W_o}^\top\left[\mathbf{x_t} , h_{t-1}\right ] +b_o\right)$.

```{figure} ./images/lstm1.png
:name: lstm1
Cellule LSTM
```

En résumé, un LSTM effectue donc les opérations suivantes à l'instant
$t$ : $$\begin{aligned}
g^{t} &=& \sigma\left(\mathbf{W_C}^\top \left[\mathbf{x_t} , h_{t-1}\right ] +b_C\right)\\
i^{t} &=& \sigma\left(\mathbf{W_i}^\top \left[\mathbf{x_t} , h_{t-1}\right ] +b_i\right)\\
f^{t} &=& \sigma\left(\mathbf{W_f}^\top \left[\mathbf{x_t} , h_{t-1}\right ] +b_f\right)\\
o^{t} &=& \sigma\left(\mathbf{W_o}^\top \left[\mathbf{x_t}, h_{t-1}\right ] +b_o\right)\\
C^{t} &=&g^{t}.i^{t} + C^{t-1}.f^{t}\\
h_t &=& o^{t} tanh(C^{t})
\end{aligned}$$

### GRU

En 2014 [@ChO14], une version simplifiée des réseaux LSTM a été
introduite, qui nécessite moins de paramètres. Les GRU (*Gated Recurrent
Units*) sont en effet des réseaux sans mémoire interne $C^{t}$, ni porte
de sortie $o^{t}$. Ces réseaux sont composés de deux portes au lieu de
trois :

-   une *porte reset* $r^{t}$, qui détermine la manière de combiner la
    nouvelle entrée au temps $t$ avec la mémoire provenant du temps
    $t-1$.

-   une *porte de mise à jour* $z^{t}$, qui détermine la quantité de
    mémoire précédente qui doit être conservée. Cette porte est la
    combinaison des portes d'entrée et d'oubli des LSTM.

Formellement : $$\begin{aligned}
r^{t} &=& \sigma\left(\mathbf{W_r}^\top \left[\mathbf{x_t} , h_{t-1}\right ] +b_r\right)\\
z^{t} &=& \sigma\left(\mathbf{W_z}^\top\left[\mathbf{x_t}, h_{t-1}\right ] +b_z\right)\\
\tilde{h}^{t} &=& tanh\left(\mathbf{W}^\top\left[\mathbf{x_t} , r^{t} h_{t-1}\right ] +b_h \right)\\ 
h_t&=&\left(1-z^{t}\right)h_{t-1} + z^{t} \tilde{h}^{t}
\end{aligned}$$

Si, pour tout $t, r^{t}=1$ et $z^{t}=0$, alors on modélise un réseau
récurrent classique.

<figure id="F:lstmC">

<figcaption>Cellule GRU</figcaption>
</figure>

### Réseaux récurrents bidirectionnels

Les réseaux bidirectionnels ont été décrits pour la première fois en
1997 [@SCH97]. Dans ces réseaux, deux couches cachées sont présentes,
chacune connectée à l'entrée et la sortie. La première couche cachée a
des connexions récurrentes depuis le passé vers le futur, tandis que
l'autre transmet les activations depuis le futur vers le passé
(figure [1.4](#F:BDRN){reference-type="ref" reference="F:BDRN"}).

<figure id="F:BDRN">
<embed src="images/bidirectionnel.pdf" />
<figcaption>Réseau bidirectionnel.</figcaption>
</figure>

Étant données une entrée et une sortie du réseau (des séquences), le
réseau peut être entraîné par rétropropagation après avoir été déplié :
$$\begin{aligned}
x_t &=&\sigma\left(\mathbf{W_h}^\top \left [\mathbf{x_t},h_{t-1} \right ]+b_h \right)\\
z_{t} &=&\sigma\left(\mathbf{W_z}^\top \left [\mathbf{x_t},z_{t+1} \right ]+b_z \right)\\
\hat{y}_{t}&=& softmax\left(\mathbf{W_y}^\top \left [x_t,z_{t} \right ]+b_y \right)
\end{aligned}$$ où $h_t$ (respectivement $z_{t}$) représente la valeur
de la couche cachée dans le sens du temps (respectivement dans le sens
inverse). Puisque le temps doit être fini dans les deux sens de
parcours, les réseaux bidirectionnels ne peuvent traiter que des
séquences finies.

### Machines de Turing neuronales

Les réseaux récurrents sont performants pour construire une
représentation implicite de l'information, mais restent relativement peu
adaptés à la conservation d'informations explicites (des dates précises
par exemple). S'inspirant des mémoires de travail, théorisées par les
neurosciences et qui sont responsables du raisonnement inductif et de la
création de nouveaux concepts, l'idée est alors d'ajouter à ces modèles
une mémoire de travail externe, ce qui permet de découpler la mémoire
(assimilable à la RAM d'un ordinateur) des opérations liées à la tâche
effectuée par le réseau (assimilable à la CPU). Puisque la mémoire des
LSTM est distribuée dans chaque cellule, elle est donc liée au nombre de
cellules et à la capacité de calcul et ce modèle ne répond pas
directement au problème posé.

Graves et al. [@Graves14] proposent alors une architecture, appelée
machine de Turing neuronale, constituée de deux éléments principaux :
une mémoire et un contrôleur doté d'un mécanisme d'attention qui lit et
écrit dans cette mémoire. Les accès mémoire sont ici des équivalents
analogiques dérivables, pour permettre d'entraîner le contrôleur par
descente de gradient. Typiquement, le contrôleur est un réseau de
neurones ou un réseau récurrent type LSTM
(figure [\[F:NTM\]](#F:NTM){reference-type="ref" reference="F:NTM"}).

Les têtes de lecture et d'écriture interagissent avec la mémoire. Chaque
tête est contrôlée par un vecteur de poids, chaque composante
définissant le degré d'interaction de la tête avec la zone mémoire
correspondante. Un *mécanisme de mise à jour de ces poids*, composé de
quatre opérations, est mis en place pour permettre l'apprentissage du
réseau :

1.  Le réseau s'intéresse tout d'abord aux zones mémoire proches d'une
    clé $k_t$ donnée. Cela permet au modèle de retrouver une information
    spécifique, en recherchant si la zone mémoire $M_t(i)$ est proche de
    la clé, au sens d'une similarité $K$. Formellement, chaque poids
    correspondant à la zone mémoire $i$ est calculé par
    $w_t(i) = softmax(\beta_t K[k_t,M_t(i)])$.

2.  Un mécanisme d'interpolation linéaire permet ensuite de mettre à
    jour les poids en fonction de leur valeur précédente (pour prendre
    plus ou moins en compte l'information issue de la clé, ou au
    contraire la valeur précédente du poids) :
    $w_t(i)=g_t.w_t(i) + (1-g_t).w_{t-1}(i)$.

3.  Un décalage par convolution translate ensuite les poids, à la
    manière du décalage classique de la tête dans une machine de Turing
    classique : $w_t(i)=\displaystyle\sum_j w_t(j)\mathbf{s_t}(i-j)$ où
    $\mathbf{s_t}$ est un vecteur qui définit un décalage des poids à
    l'instant $t$.

4.  Enfin, le vecteur de poids est focalisé :
    $w_t(i) = w_t(i)^{\gamma_t}$, $\gamma_t>1$.

Une fois que la tête a mis à jour les poids, elle interagit avec la
mémoire :

-   Dans le cas de la tête de lecture, elle calcule une combinaison
    linéaire des zones mémoire, pondérées par les poids $w_t(i)$, et
    produit le vecteur $\mathbf{r_t}$, fourni au contrôleur de l'instant
    suivant.

-   Dans le cas de la tête d'écriture, le contenu de la mémoire est mis
    à jour selon la formule
    $M_t(i) = M_{t-1}(i)(1-w_t(i)\mathbf{e_t})+w_t(i)\mathbf{a_t}$, où
    $\mathbf{e_t}$ est un vecteur d'effacement, dont les composantes
    sont dans {0,1} et $\mathbf{a_t}$ est un vecteur d'ajout.

::: SCfigure
![image](images/turing.pdf){width=".6\\linewidth"}
:::

<figure>

<figcaption>Quelques applications des réseaux récurrents.</figcaption>
</figure>

## Quelques applications

Comme les réseaux convolutifs, les réseaux récurrents ont depuis leur
introduction trouvé de nombreuses applications.

**Traitement automatique du langage**

Les réseaux récurrents sont utilisés en traitement automatique du
langage, notamment à des fins génératives. Ces réseaux permettent de
modéliser un langage (prédire la probabilité d'un mot donné étant donnés
les mots précédents) et de générer du texte à partir du modèle appris.
De nombreuses applications découlent de cette modélisation : génération
de texte au style de (génération d'un texte dans le style de
Shakespeare, à partir d'un RNN appris sur le corpus des œuvres de
l'auteur par exemple, génération de textes manuscrits
(figure [\[F:gener\]](#F:gener){reference-type="ref"
reference="F:gener"}), génération de pages Wikipedia, ou même génération
d'articles scientifiques, à partir des sources LaTeXd'un ouvrage et d'un
LSTM multicouche.

**Traduction automatique**

La traduction automatique de texte procède de la même stratégie que la
modélisation d'une langue. Deux réseaux récurrents sont entraînés,
chacun dans une des langues, et le RNN traducteur calcule sa sortie en
fonction de la couche cachée du premier réseau.

**Analyse de sentiments**

Détecter de manière automatique l'opinion du public sur un sujet donné
intéresse de plus en plus le domaine commercial. Ce domaine, largement
alimenté par les réseaux sociaux, les avis et recommandations déposées
sur les sites Internet, est un champ de prédilection pour les réseaux
profonds. Des réseaux récurrents (notamment LSTM structurés en arbres)
sont utilisés à cet effet et servent de base à des systèmes de
recommandation.

**Résumé automatique**

Les réseaux récurrents permettent de produire des résumés abstraits de
textes (i.e. générer de nouvelles phrases, en opposition à extraire les
mots les plus importants d'un texte). Les modèles utilisés sont des
réseaux récurrents avec mécanisme d'attention. Un système
d'encodage/décodage est mis en place dans le réseau, où l'encodeur est
par exemple un GRU bidirectionnel et le décodeur un GRU dont l'état
caché a la même taille que celui de l'encodeur. Les modèles sont appris
et validés sur des corpus dédiés (DUC, CNN/Daily Mail par exemple).

**Reconnaissance de la parole**

L'utilisation de réseaux LSTM bidirectionnels, qui permettent à la fois
d'exploiter les contextes passé et futur, et de garder trace d'un
contexte à longue échéance, a montré de bonnes performances dans la
tache de reconnaissance de la parole.

**Annotation d'images**

Couplé à un réseau convolutif, un RNN permet de générer des descriptions
(légendes) d'images non labelisées. Le réseau convolutif produit des
descripteurs, qui servent d'entrée à un réseau récurrent type LSTM
(figure [\[F:caption\]](#F:caption){reference-type="ref"
reference="F:caption"}).



```{bibliography}
:style: unsrt
```
