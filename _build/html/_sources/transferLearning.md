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
# Utilisation de réseaux existants

Nous présentons dans la suite quatre réseaux profonds classiques. Nous montrons ensuite comment les utiliser directement, ou comment les adapter pour répondre à une problématique précise, en lien avec leur utilisation originale ou non. Nous introduisons enfin une manière
d'apprendre un réseau à partir de peu de données.

## Quelques réseaux profonds classiques

Les réseaux présentés ici ont prouvé leur efficacité, notamment lors des compétitions organisées depuis 2010 sur une base de données d'images nommée [ImageNet](http://www.image-net.org/). Initiée à l'Université de Stanford, cette base de données comporte aujourd'hui plus de 14 millions d'images, classées en 21841 catégories (avions, voitures, chats,...). Dans les compétitions [ILSVRC](https://image-net.org/challenges/LSVRC/) ( ImageNet Large Scale Visual Recognition Challenge), les chercheurs se voient proposer
une extraction de 1,2 millions d'images d'entraînement, 100 000 images de test et 50 000 images de validation, catégorisées en 1000 classes. Le gagnant est celui qui atteint la meilleure précision de reconnaissance sur les 5 premières classes (top-5).

La {numref}`perf` donne un aperçu des performances de plusieurs réseaux profonds suivant cette métrique.


```{figure} ./images/classifImagenet.png
:name: perf
Performance de réseaux profonds sur une tache de classification ([source](https://theaisummer.com/cnn-architectures/)
```


### AlexNet

En 2012, Krizhevsky et al  {cite:p}`Krizhevsky12` remportent ILSVRC avec un taux de reconnaissance de 84.6%, en utilisant AlexNet, un réseau convolutif composé de 5 couches de convolution et de pooling, suivies de 3 couches complètement connectées ({numref}`alexnet`).


```{figure} ./images/AlexNet.png
:name: alexnet
Architecture du réseau AlexNet. Les couches de convolution
et d’activation sont en orange clair, les couches d’agrégation en orange
foncé. Les couches complètement connectées sont en violet.
```

Si la profondeur du réseau reste faible, le nombre de paramètres était
déjà important. En regardant uniquement la première couche de
convolution, on constate que :

-   l'entrée est composée d'images 227$\times$227$\times$3
-   les filtres de convolution sont de taille 11
-   le pas de convolution (stride) est de 4

Ainsi la sortie de la couche de convolution est de taille 55$\times$55$\times$96=290 400 neurones, chacun ayant
11$\times$11$\times$3=363 poids et un biais. Cela implique, sur cette couche de convolution seulement, 105 705 600 paramètres à ajuster.

Ce réseau, amélioration d'un réseau existant (LeNet), apportait de
nombreuses contributions, comme l'utilisation de couches ReLU, de
dropout, ou du GPU (NVIDIA GTX 580) pendant la phase d'entraînement.

### VGG

Les réseaux VGG (Visual Geometry Group, université d'Oxford)
{cite:p}`Simonyan14` ont été les premiers réseaux à utiliser de petits filtres de convolution (3$\times$3) et à les combiner pour
décrire des séquences de convolution, l'idée étant d'émuler l'effet de
larges champs réceptifs par cette séquence. Cette technique amène
malheureusement à un nombre exponentiel de paramètres (le modèle
entraîné qui peut être téléchargé a une taille de plus de 500 Mo). VGG a concouru à ILSVRC 2014, a obtenu un taux de bonne classification de
92.3% mais n'a pas remporté le challenge. Aujourd'hui VGG et une famille de réseaux profonds (de A à E) qui varient par leur architecture

::: center
<figure id="F:VGG">
<style type="text/css">
.myTable {style="border:1px solid blue; border-collapse:collapse;" }
.myTable th { background-color:#000;color:white;width:50%; }
.myTable td, .myTable th { padding:5px;border:1px solid #000; }
</style>

<table class="myTable">
<thead>
<tr class="header">
<th style="text-align: center;"><strong>A</strong></th>
<th style="text-align: center;"><strong><span>A-LRN</span></strong></th>
<th style="text-align: center;"><strong><span>B</span></strong></th>
<th style="text-align: center;"><strong><span>C</span></strong></th>
<th style="text-align: center;"><strong><span>D</span></strong></th>
<th style="text-align: center;"><strong><span>E</span></strong></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: center;">11 couches</td>
<td style="text-align: center;">11 couches</td>
<td style="text-align: center;">13 couches</td>
<td style="text-align: center;">16 couches</td>
<td style="text-align: center;">16 couches</td>
<td style="text-align: center;">19 couches</td>
</tr>
<tr class="even">
<td colspan="6" style="text-align: center;">Entrée : image 224<span
class="math inline">×</span>224 RGB</td>
</tr>
<tr class="odd">
<td rowspan="2" style="text-align: center;"></td>
<td style="text-align: center;">conv3-64</td>
<td style="text-align: center;">conv3-64</td>
<td style="text-align: center;">conv3-64</td>
<td style="text-align: center;">conv3-64</td>
<td style="text-align: center;">conv3-64</td>
</tr>
<tr class="even">
<td style="text-align: center;">LRN</td>
<td style="text-align: center;">conv3-64</td>
<td style="text-align: center;">conv3-64</td>
<td style="text-align: center;">conv3-64</td>
<td style="text-align: center;">conv3-64</td>
</tr>
<tr class="odd">
<td colspan="6" style="text-align: center;">max pooling</td>
</tr>
<tr class="even">
<td rowspan="2" style="text-align: center;"></td>
<td style="text-align: center;">conv3-128</td>
<td style="text-align: center;">conv3-128</td>
<td style="text-align: center;">conv3-128</td>
<td style="text-align: center;">conv3-128</td>
<td style="text-align: center;">conv3-128</td>
</tr>
<tr class="odd">
<td style="text-align: center;"></td>
<td style="text-align: center;">conv3-128</td>
<td style="text-align: center;">conv3-128</td>
<td style="text-align: center;">conv3-128</td>
<td style="text-align: center;">conv3-128</td>
</tr>
<tr class="even">
<td colspan="6" style="text-align: center;">max pooling</td>
</tr>
<tr class="odd">
<td rowspan="4" style="text-align: center;"></td>
<td style="text-align: center;">conv3-256</td>
<td style="text-align: center;">conv3-256</td>
<td style="text-align: center;">conv3-256</td>
<td style="text-align: center;">conv3-256</td>
<td style="text-align: center;">conv3-256</td>
</tr>
<tr class="even">
<td style="text-align: center;">conv3-256</td>
<td style="text-align: center;">conv3-256</td>
<td style="text-align: center;">conv3-256</td>
<td style="text-align: center;">conv3-256</td>
<td style="text-align: center;">conv3-256</td>
</tr>
<tr class="odd">
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;">conv1-256</td>
<td style="text-align: center;">conv3-256</td>
<td style="text-align: center;">conv3-256</td>
</tr>
<tr class="even">
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;">conv3-256</td>
</tr>
<tr class="odd">
<td colspan="6" style="text-align: center;">max pooling</td>
</tr>
<tr class="even">
<td rowspan="4" style="text-align: center;"></td>
<td style="text-align: center;">conv3-512</td>
<td style="text-align: center;">conv3-512</td>
<td style="text-align: center;">conv3-512</td>
<td style="text-align: center;">conv3-512</td>
<td style="text-align: center;">conv3-512</td>
</tr>
<tr class="odd">
<td style="text-align: center;">conv3-512</td>
<td style="text-align: center;">conv3-512</td>
<td style="text-align: center;">conv3-512</td>
<td style="text-align: center;">conv3-512</td>
<td style="text-align: center;">conv3-512</td>
</tr>
<tr class="even">
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;">conv1-512</td>
<td style="text-align: center;">conv3-512</td>
<td style="text-align: center;">conv3-512</td>
</tr>
<tr class="odd">
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;">conv3-512</td>
</tr>
<tr class="even">
<td colspan="6" style="text-align: center;">max pooling</td>
</tr>
<tr class="odd">
<td rowspan="4" style="text-align: center;"></td>
<td style="text-align: center;">conv3-512</td>
<td style="text-align: center;">conv3-512</td>
<td style="text-align: center;">conv3-512</td>
<td style="text-align: center;">conv3-512</td>
<td style="text-align: center;">conv3-512</td>
</tr>
<tr class="even">
<td style="text-align: center;">conv3-512</td>
<td style="text-align: center;">conv3-512</td>
<td style="text-align: center;">conv3-512</td>
<td style="text-align: center;">conv3-512</td>
<td style="text-align: center;">conv3-512</td>
</tr>
<tr class="odd">
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;">conv1-512</td>
<td style="text-align: center;">conv3-512</td>
<td style="text-align: center;">conv3-512</td>
</tr>
<tr class="even">
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;">conv3-512</td>
</tr>
<tr class="odd">
<td colspan="6" style="text-align: center;">max pooling</td>
</tr>
<tr class="even">
<td colspan="6" style="text-align: center;">Couche complètement
connectée 4096 neurones</td>
</tr>
<tr class="odd">
<td colspan="6" style="text-align: center;">Couche complètement
connectée 4096 neurones</td>
</tr>
<tr class="even">
<td colspan="6" style="text-align: center;">Couche complètement
connectée 1000 neurones</td>
</tr>
<tr class="odd">
<td colspan="6" style="text-align: center;">Classifieur softmax</td>
</tr>
</tbody>
</table>
</figure>
:::


Le nombre de paramètres (en millions) pour les réseaux de A à E est 133, 133, 134, 138 et 144. Les réseaux VGG-D et VGG-E sont les plus précis et populaires.


```{figure} ./images/VGG16.png
:name: vgg16
Réseau VGG16
```

### Inception

Inception, proposé par Google, est le premier réseau dont les
performances ont été augmentées non seulement en augmentant le nombre de couches, mais en pensant et optimisant le design et l'architecture.
L'idée est ici d'utiliser plusieurs filtres, de tailles différentes, sur la même image et de concaténer les résultats pour générer une
représentation plus robuste.

Inception n'est pas un réseau, c'est une famille de réseaux : Network in Network {cite:p}`Lin13`, Inception V1 {cite:p}`Szegedy14`, Inception V2 {cite:p}`Szegedy15`,
Xception {cite:p}`Chollet16`,...

L'idée du premier réseau ({numref}`nin`) est de connecter les couches de convolution par des perceptrons multicouches, introduisant des non linéarités dans les réseaux profonds. Mathématiquement, ces perceptrons sont équivalents à des convolutions par des filtres 1$\times 1$ et gardent donc la cohérence des réseaux. Cette nouvelle architecture rend moins indispensable les couches complètement connectées en fin de réseau. Les auteurs moyennent spatialement les cartes finales et donnent le résultat au classifier softmax. Le nombre de paramètres est alors réduit, diminuant de ce fait le risque de sur apprentissage.


```{figure} ./images/NIN.png
:name: nin
Réseau Network in Network
```

Inception V1, implémenté dans le réseau GoogLeNet vainqueur d'ILSVRC
2014, est une extension à des réseaux plus profonds de Network to
Network. Le réseau est composé de 22 couches et atteint 93.3% de taux de reconnaissance. D'autres améliorations théoriques (fonctions de pertes associées aux couches intermédiaires dans la phase d'apprentissage, introduction de caractères épars dans le réseau) ont également permis d'améliorer les performances (de calcul et de classification).

Inception V2, puis V3 ({numref}`inceptionv3`) adoptent des techniques de factorisation (toute convolution par un filtre de taille plus grande que 3$\times$ 3 peut être exprimée de manière plus efficace
avec une série de filtres de taille réduite) et de normalisation pour
améliorer encore les performances.

Inception V4 {cite:p}`Szegedy16` propose une version rationalisée, à
l'architecture uniforme et aux performances accrues.

```{figure} ./images/inceptionv3.png
:name: inceptionv3
Architecture d’inception V3
```


### ResNet

En 2015, Microsoft remporte la compétition ILSVRC avec ResNet {cite:p}`He15`, un réseau à 152 couches qui utilise un module ResNet. Le taux de bonne reconnaissance est de 96.4%. Un réseau résiduel (ou ResNet) résout le problème de vanishing gradient de la manière la plus simple possible, en permettant des raccourcis entre chaque couche du réseau. Dans un réseau classique, l'activation en sortie de couche est de la forme $y=\sigma(x)$, et lors de la rétropropagation, le gradient doit nécessairement repasser par $\sigma(x)$, ce qui peut causer des
problèmes en raison de la (forte) non linéarité induite par $\sigma$.
Dans un réseau résiduel, la sortie de chaque couche est calculée par
$y=\sigma()+x$, où $+x$ est le raccourci entre chaque couche, qui permet
au gradient de transiter directement sans passer par $\sigma$.\
Cette représentation donne l'idée générale, mais la réalité est un peu
plus complexe, et prend la forme d'un module ResNet ({numref}`resnet`).


```{figure} ./images/resNet.png
:name: resnet
Module ResNet (source : {cite:p}`He15`)
```

## Comment utiliser ces réseaux ?

Il est possible de définir les réseaux classiques en décrivant une à une les couches et leur paramètres, qui sont proposés dans les articles correspondants. On imagine assez bien le travail que cela peut représenter sur ResNet par exemple...

Fort heureusement, il existe d'autres manières d'utiliser ces réseaux.

### Utilisation de réseaux pré-entraînés

Il est possible de charger / sauvegarder des
réseaux qui ont été entraînés sur des grandes bases de données et de les utiliser directement. Il est également possible, pendant
l'entraînement, de créer des sauvegardes (checkpoints) pour reprendre
éventuellement l'entraînement en cours d'itérations. On peut sauvegarder
tout le réseau (architecture + optimiseur + poids), ou seulement les
poids.

`torchvision`donne accès à de nombreux modèles pré-entrainés :  

```{code-cell} ipython3
from torchvision import models
dir(models)
```

Il est alors facile de charger un tel réseau, par exemple

```python
resnet = models.resnet101(pretrained=True)
```


### Transfer learning et fine tuning

Il est possible d'utiliser les réseaux classiques pré-entrainés pour de
nouvelles tâches. L'idée sous-jacente et que les premières couches
capturent des caractéristiques bas niveau, et que la sémantique vient
avec les couches profondes. Ainsi, dans un problème de classification,
où les classes n'ont pas été apprises, on peut supposer qu'en conservant
les premières couches on extraira des caractéristiques communes des
images, et qu'en changeant les dernières couches (information sémantique
et haut niveau et étage de classification), c'est à dire en réapprenant
les connexions, on spécifiera le nouveau réseau pour la nouvelle tâche
de classification.\
Cette approche rentre dans le cadre des méthodes de Transfer Learning
[@Pan10] et de fine tuning, cas particulier d'adaptation de domaine :

-   les méthodes de transfer learning prennent un réseau déjà entraîné,
    enlèvent la dernière couche complètement connectée, et traitent le
    réseau restant comme un extracteur de caractéristiques. Un nouveau
    classifieur est alors entraîné sur les caractéristiques calculées
    sur le nouveau problème

-   les méthodes de fine tuning ré-entrainent le classifieur du réseau,
    et remettent à jour les poids du réseau pré-entraîné par
    rétropropagation.

Plusieurs facteurs influent sur le choix de la méthode à utiliser : la
taille des données d'apprentissage du nouveau problème, et la
ressemblance du nouveau jeu de données avec celui qui a servi à
entraîner le réseau initial :

-   pour un jeu de données similaire de petite taille, on utilise du
    transfer learning, avec un classifieur utilisé sur les
    caractéristiques calculées sur les dernières couches du réseau
    initial

-   pour un jeu de données de petite taille et un problème différent, on
    utilise du transfer learning, avec un classifieur utilisé sur les
    caractéristiques calculées sur les premières couches du réseau
    initial

-   pour un jeu de données, similaire ou non de grande taille, on
    utilise le fine tuning

A noter qu'il est toujours possible d'augmenter la taille du jeu de
données par des technique de \"Data Augmentation\" (changement de
couleurs des pixels, rotations, cropping, homothéties, translations\...)

## Que faire si j'ai peu de données ?

Les méthodes supervisées nécessitent pour de bonnes performances un
ensemble d'apprentissage $\mathcal{S}$ de grand cardinal. Si seulement
peu d'exemples
$\mathcal{S}_s = \{(\mathbf{x}_i,y_i), i\in[\![1\cdots m ]\!]\}$ sont
disponibles, avec $m$ petit, les techniques précédemment décrites ne
sont pour la plupart plus applicables.\
Les méthodes de Few-Shot Learning ont été introduites pour traiter ce
manque de données. Les exemples applicatifs sont nombreux, allant de la
classification d'images à l'analyse de sentiments à partir de textes, ou
encore à la reconnaissance d'objets.\
Vu sous l'angle de la minimisation du risque empirique, l'hypothèse $h$
construite sur la minimisation de
$$R(h) = \displaystyle\sum_{i=1}^m \ell(y_i,h(\mathbf{x}_i))$$ conduit à
un sur apprentissage et un risque $R(h)$ très loin du risque réel. Pour
pallier ce problème, des connaissances *a priori* doivent être
utilisées. Le Few-shot learning propose trois alternatives. Nous
détaillons ici l'une d'entre elles, l'augmentation de données.\
Les approches de cette catégorie utilisent des connaissances *a priori*
sur les données pour enrichir $\mathcal{S}_s$. On les regroupe parfois
sous le vocable de méthodes d'*augmentation de données*. Si elles sont
faciles à mettre en oeuvre et à comprendre, ces méthodes restent
cependant dépendantes du domaine d'étude et ne peuvent être facilement
généralisées.

Les principales stratégies sont résumées dans le tableau
[\[T:dataAug\]](#T:dataAug){reference-type="ref" reference="T:dataAug"}
et un exemple d'illustration est donné figure
[1.10](#Fig:dataAug){reference-type="ref" reference="Fig:dataAug"}.

::: tabular
M4cm\|M2.5cm\|M5cm\|M2.5cm **Transformation\... & **Entrée & **Opérateur
& **Sortie\
********

\... de données de $\mathcal{S}_s$

& $(\mathbf{x}_i,y_i)\in \mathcal{S}_s$ &
$t:\mathcal{X}\rightarrow \mathcal{X}$ & $(\mathbf{t(x_i)},y_i)$\

\... d'un ensemble de données non étiquetées

& $(\mathbf{x},-)$ &

$h:\mathcal{X}\rightarrow \mathcal{Y}$ entraîné sur $\mathcal{S}_s$

& $(\mathbf{x},h(\mathbf{x}))$\

\... d'un ensemble de données similaires

& $\{(\mathbf{\hat{x}_j},\hat{y}_j)\}$ &

Opérateur de combinaison $c$

& $(c(\{\mathbf{\hat{x}_j}\}),c(\{\hat{y}_j\}))$\
:::

<figure id="Fig:dataAug">
<img src="images/DataAugmentation" />
<figcaption>Exemple d’augmentation de données. De gauche à droite :
image originale, rotation de 20<span
class="math inline"><sup>∘</sup></span>, flip, ajout de bruit gaussien,
déformation élastique, changement de contraste par canal
RGB.</figcaption>
</figure>



```{bibliography}
:style: unsrt
```