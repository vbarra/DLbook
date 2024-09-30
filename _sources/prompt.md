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

# Ingénierie du prompt

L'ingénierie des messages, également connue sous le nom de « In-Context Prompting », se réfère aux méthodes de communication avec le LLM afin d'orienter son comportement vers les résultats souhaités sans mettre à jour les données.
communiquer avec LLM pour orienter son comportement vers les résultats souhaités sans mettre à jour les poids du modèle.
sans mettre à jour les poids du modèle. Il s'agit d'une science empirique et l'effet des méthodes d'ingénierie d'invite
peut varier considérablement d'un modèle à l'autre, ce qui nécessite de nombreuses expériences et des heuristiques.


## Méthodes de base

### Zero-shot
La méthode zero-shot consiste simplement à transmettre le texte de la tâche au modèle et à demander des résultats. Dans l'analyse de sentiments, par exemple, on transmet juste le texte au LLM, en attendant une réponse de type positif/négatif (voire neutre pour certains modèles).



### Few shot
La méthode few shot présente un ensemble d'exemples d'entraînement de haute qualité, chacun comprenant à la fois l'entrée et la sortie souhaitée, sur la tâche cible. Au fur et à mesure que le modèle voit de bons exemples, il peut mieux comprendre l'intention humaine et les critères relatifs aux types de réponses souhaitées. 
En contrepartie, le modèle utilise un nombre de tokens plus important et peut se heurter à la limite de longueur du contexte lorsque les textes d'entrée et de sortie sont longs.
De nombreuses études se sont penchées sur la manière de construire des exemples pour maximiser les performances et ont observé que le choix du format de l'invite, des exemples d'entraînement et de l'ordre des exemples peut conduire à des performances très différentes, allant d'une de la devinette aléatoire à une performance égalant l'état de l'art.

Pour sélectionner les exemples, on peut par exemple :
- choisir des exemples qui sont sémantiquement similaires à l'exemple test en utilisant le regroupement dans l'espace latent.
- sélectionner un ensemble d'exemples variés et représentatifs à l'aide d'une approcha basée graphe. Pour ce faire, on construit un graphe orienté $G=(V,E)$ fondé sur une mesure de similarité cosinus entre la représentation des exemples dans l'espace latent du LLM, chaque noeud du graphe étant en relation avec ses $k$ plus proches voisins. A partir d'un ensemble d'exemples sélectionnés $\mathcal L$ initialement vide et d'un ensemble d'exemples restants $\mathcal U$, on affecte un score à chaque $u\in\mathcal U$ par 

$$ score(u) = \displaystyle\sum_{v\in \{v, (u,v)\in E, v\in\mathcal U\}}\rho^{|l\in\mathcal L, (v,l)\in E|}, \rho>1$$

le terme dans la somme, dépendant de $v$, est petit si un grand nombre de voisins de $v$ sont sélectionnés , le score encourageant alors la diversité des exemples.  

Pour ordonner les exemples, une suggestion générale est de garder la sélection d'exemples diversifiée, pertinente pour l'échantillon de test et dans un ordre aléatoire afin d'éviter le biais de l'étiquette majoritaire.

L'augmentation de la taille des modèles ou l'inclusion d'un plus grand nombre d'exemples d'apprentissage ne réduit pas la variance entre les différentes permutations d'exemple. Le même ordre peut bien fonctionner pour un modèle mais mal pour un autre. Lorsque l'ensemble de validation est limité, il faut envisager de choisir l'ordre de manière à ce que le modèle ne produise pas de prédictions déséquilibrées ou qu'il ne soit pas trop confiant dans ses prédictions.


## Prompt par instruction

