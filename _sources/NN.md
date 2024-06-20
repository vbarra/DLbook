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

# Les réseaux de neurones

## Réseaux de neurones et apprentissage automatique

Les réseaux de neurones artificiels sont des techniques
issues du domaine du connexionisme. Le courant connexionniste insiste
sur le grand nombre de connexions (sous forme de réseau) réalisées entre
les différents automates que sont les neurones. Le connexionisme permet
:

-   de disposer de nouveaux moyens de calcul : conversion de
    l'information des systèmes avec des applications pratiques ;

-   de modéliser des phénomènes biologiques pour en apprendre davantage
    sur le cerveau en l'observant comme si c'était une machine de
    traitement électrique.

La démarche des réseaux de neurones s'oppose en certains points à celle
de l'intelligence artificielle basée règles,
qui sont manipulées selon les techniques de la logique formelle afin de
fournir une représentation explicite du raisonnement. Cette méthodologie implique
une approche \"descendante\" : elle part de l'analyse de la manière
dont l'être humain procède pour résoudre des problèmes ou pour les
apprendre, et tente de restituer cette démarche en la décomposant en
unités élémentaires. 
Les réseaux de neurones, eux, procèdent selon une
approche \"ascendante\" qui tente de produire des phénomènes complexes
à partir
d'opérations très élémentaires.

### Du neurone biologique au neurone formel

La reconnaissance du fait que le cerveau fonctionne de manière
entièrement différente de celle d'un ordinateur conventionnel a joué un
rôle très important dans le développement des réseaux de neurones
artificiels. Les travaux effectués pour essayer de comprendre le
comportement du cerveau humain ont mené à représenter celui-ci par un
ensemble de composants structurels appelés neurones, massivement
interconnectés entre eux. Le cerveau humain en contient en moyenne une dizaine de milliards, chacun d'entre eux étant connecté, encore une fois en moyenne,
connecté à dix mille autres.\
Le neurone biologique est composé de quatre parties distinctes ({numref}`neurone`) :

-   le *corps cellulaire*, qui contient le noyau de la cellule nerveuse;
    c'est en cet endroit que prend naissance l'influx nerveux, qui
    représente l'état d'activité du neurone;

-   les *dendrites*, ramifications tubulaires courtes formant une espèce
    d'arborescence autour du corps cellulaire; ce sont les entrées
    principales du neurone, qui captent l'information venant d'autres
    neurones;

-   l'*axone*, longue fibre nerveuse qui se ramifie à son extrémité; c'est
    la sortie du neurone et le support de l'information vers les autres
    neurones;

-   la *synapse*, qui communique l'information, en la pondérant par un
    poids synaptique, à un autre neurone; elle est essentielle dans le
    fonctionnement du système nerveux.

```{figure} ./images/neurone.png
:name: neurone
Neurone biologique
```



La transmission de l'information d'un neurone à l'autre s'effectue au
moyen de l'influx nerveux, qui est constitué d'une impulsion électrique,
d'une durée d'environ 2 ms et d'une amplitude de 100 mV. Une cellule
nerveuse standard non sollicitée en émet en moyenne cinquante à la
seconde (activité spontanée). La probabilité d'émettre une impulsion est
accrue ou réduite selon que la somme pondérée des entrées du neurone est
globalement excitatrice ou inhibitrice. Cette fréquence peut ainsi être
portée jusqu'à 100 impulsions par seconde, pour un neurone bombardé
d'effets synaptiques excitateurs ; dans le cas contraire, elle peut être
réduite à néant (le neurone reste silencieux). Les effets synaptiques
qui agissent sur le neurone entraînent donc une modulation de la
fréquence d'émission de l'influx nerveux. Le message transmis est
précisément contenu dans le nombre d'influx nerveux émis, défini par une
moyenne sur quelques dizaines de ms. L'information contenue dans le
cerveau, quant à elle, est représentée par les poids synaptiques
attribués aux entrées de chaque neurone. Le cerveau est capable
d'organiser ces neurones, selon un assemblage complexe, non-linéaire et
extrêmement parallèle, de manière à pouvoir accomplir des tâches très
élaborées. Du fait du grand nombre de neurones et de leurs
interconnexions, ce système possède une propriété de tolérance aux
fautes. Ainsi, la défectuosité d'un neurone n'entraînera aucune perte
réelle d'information, mais seulement une faible dégradation en qualité
de toute l'information contenue dans le système. 

C'est la tentative de donner à l'ordinateur les qualités de perception
du cerveau humain qui a conduit à une modélisation électrique de
celui-ci. C'est cette modélisation que tentent de réaliser les réseaux
de neurones artificiels, dont l'élaboration repose sur base de la
définition suivante, proposée par Haykin :

Un réseau de neurones est un processus distribué de manière massivement
parallèle, qui a une propension naturelle à mémoriser des connaissances
de façon expérimentale et de les rendre disponibles pour utilisation. Il
ressemble au cerveau en deux points :

1.  la connaissance est acquise au travers d'un processus
    d'apprentissage;

2.  les poids des connections entre les neurones sont utilisés pour
    mémoriser la connaissance

La première étude systématique du neurone artificiel est due au
neuropsychiatre McCulloch et au logicien Pitts qui s'inspirèrent de
leurs travaux sur les neurones biologiques.

### Classification des réseaux de neurones

Un réseau de neurones est constitué d'un grand nombre de
cellules de base interconnectées. De nombreuses variantes sont définies
selon le choix de la cellule élémentaire, de l'architecture et de la
dynamique du réseau.

Une cellule élémentaire peut manipuler des valeurs binaires ou réelles.
Les valeurs binaires sont représentées par 0 et 1 ou -1 et 1.
Différentes fonctions d'ctivation peuvent être utilisées pour le calcul de la
sortie. Le calcul de la sortie peut être déterministe ou probabiliste.

L'architecture du réseau peut être sans rétroaction, c'est à dire que la
sortie d'une cellule ne peut influencer son entrée. Elle peut être avec
rétroaction totale ou partielle.

La dynamique du réseau peut être synchrone : toutes les cellules
calculent leurs sorties respectives simultanément. La dynamique peut
être asynchrone. Dans ce dernier cas, on peut avoir une dynamique
asynchrone séquentielle : les cellules calculent leurs sorties chacune à
son tour en séquence ou avoir une dynamique asynchrone aléatoire.

Par exemple, si on considère des neurones à sortie stochastique -1 ou 1
calculée par une fonction à seuil basée sur la fonction sigmoïde, une
interconnection complète et une dynamique synchrone, on obtient le
modèle de Hopfield et la notion de mémoire associative.

Si on considère des neurones déterministes à sortie réelle calculée à
l'aide de la fonction sigmoïde, une architecture sans rétroaction en
couches successives avec une couche d'entrée et une couche de sortie,
une dynamique asynchrone séquentielle, on obtient le modèle du
Perceptron multi-couches (PMC).

### Applications

En apprentissage, les réseaux de neurones sont essentiellement utilisés pour :

-   l'apprentissage supervisé ;

-   l'apprentissage non supervisé ;

-   l'apprentissage par renforcement.

Dans la suite, nous nous intéressons essentiellement au cas de l'apprentissage
supervisé. Le cas des réseaux de neurones en apprentissage non supervisé
concerne principalement les cartes de Kohonen, les machines de Boltzmann
restreintes (RBM) et les autoencodeurs.
