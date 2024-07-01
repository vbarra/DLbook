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

# LLM
Les *grands modèles de langage* (large Language Models, LLM) sont des modèles entraînés par pré-apprentissage de connaissances sur le langage et le monde à partir de grandes quantités de texte. Les grands modèles de langage affichent des performances remarquables dans toutes sortes de tâches liées au langage naturel grâce aux connaissances qu'ils acquièrent lors du pré-apprentissage. Ils ont particulièrement transformé les tâches de production de texte, comme le résumé, la traduction automatique, la réponse aux questions ou encore les chatbots.

## Rappels sur les transformers
L'architecture standard pour la construction de ces LLM est le [transformer](./transformers.md). Pour rappel, un transformer, utilisé sur du texte, prend en entrée une séquence de mots et calcule une prédiction du mot suivant, ainsi qu'une suite de valeurs (embeddings) qui modélise la signification contextuelle de chacun des mots de l'entrée.

Les transformers sont constitués de piles de blocs  ({numref}`transformer`), chacun d'entre eux étant un réseau multicouche qui fait correspondre des séquences de vecteurs d'entrée $(\boldsymbol x_1,\cdots \boldsymbol x_n)$ à des séquences de vecteurs de sortie de même longueur. Ces blocs sont constitués par la combinaison de couches linéaires, de perceptrons multicouches et de couches d'auto-attention, qui permettent à un réseau d'extraire et d'utiliser directement des informations à partir de contextes arbitrairement larges. 

