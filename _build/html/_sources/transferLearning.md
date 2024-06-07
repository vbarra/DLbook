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
Performance de réseaux profonds sur une tache de classification ([source](https://theaisummer.com/cnn-architectures/))
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
où les classes n'ont pas été apprises, on peut supposer qu'en conservant les premières couches on extraira des caractéristiques communes des
images, et qu'en changeant les dernières couches (information sémantique
et haut niveau et étage de classification), c'est à dire en réapprenant
les connexions, on spécifiera le nouveau réseau pour la nouvelle tâche
de classification.

Cette approche rentre dans le cadre des méthodes d'apprentissage par transfert (Transfer Learning)
{cite:p}`Pan10` et de fine tuning, cas particulier d'adaptation de domaine.

L'apprentissage par transfert comporte généralement deux étapes principales :

- **Extraction des caractéristiques** : dans cette étape,le modèle pré-entraîné est utilisé comme un extracteur de caractéristiques fixes. On supprime les couches finales (MLP, responsable de la classification) et on les remplaçe par de nouvelles couches spécifiques à la tâche adressée ({numref}`tl2`). Les poids du modèle pré-entraîné sont gelés et seuls les poids des couches nouvellement ajoutées sont entraînés sur l'ensemble de données du problème.

```{figure} ./images/tl2.png
:name: tl2
Réentrainement d'un classifieur sur des caractéristiques extraites.
```


- **fine tuning** : le fine tuning pousse le processus un peu plus loin en dégelant certaines des couches du modèle pré-entraîné et en leur permettant d'être mises à jour avec le nouvel ensemble de données. Cette étape permet au modèle de s'adapter et d'apprendre des caractéristiques plus spécifiques liées à la nouvelle tâche ou au nouveau domaine.


Plusieurs facteurs influent sur le choix de la méthode à utiliser, parmi lesquels : 

- la taille des données d'apprentissage du nouveau problème ({numref}`tl`) 

```{figure} ./images/tl.png
:name: tl
Stratégies d'apprentissage par transfert.
```

- la ressemblance du nouveau jeu de données avec celui qui a servi à
entraîner le réseau initial ({numref}`domaintask`).

```{figure} ./images/domaintask.png
:name: domaintask
Changement de domaine / tâche.
```

Pour un jeu de données similaire de petite taille, on utilise du
    transfer learning, avec un classifieur utilisé sur les
    caractéristiques calculées sur les dernières couches du réseau
    initial. Pour un jeu de données de petite taille et un problème différent, on utilise du transfer learning, avec un classifieur utilisé sur les caractéristiques calculées sur les premières couches du réseau initial. Pour un jeu de données, similaire ou non de grande taille, on
    utilise le fine tuning

A noter que, si peu de données sont disponibles sur la nouvelle tâche/le nouveau domaine, il est toujours possible :
- d'augmenter la taille du jeu de données par des technique de \"Data Augmentation\" (changement de
couleurs des pixels, rotations, cropping, homothéties, translations\...) ({numref}`dataaugment`).

```{figure} ./images/dataaugment.png
:name: dataaugment
Augmentation de données : à partir d'un exemple (image de gauche), on construit plusieurs autres exemples par rotation, flip, ajout de bruit, déformation, changement colorimétrique.
```

- d'utiliser plus généralement des méthodes de Few shot / Zero shot learning {cite:p}`Song23`, dont l'augmentation de données est un exemple.

## Implémentation

On propose ici d'implémenter deux stratégies : 
- une première d'entraînement d'un réseau en initialisant les poids à ceux du même réseau préentraîné sur ImageNet. Une couche de classification spécifique au problème est ajoutée, et tout le réseau est entraîné.
- une seconde ({numref}`tl2`) qui remplace le réseau de classification du réseau préentraîné par un nouveau réseau de classification, dont les poids sont entrâinés sur la nouvelle tâche. Les poids du réseau initial (hors couche de classification) sont conservés (le réseaux convolutif agit donc comme un extracteur de caractéristiques)

L'objectif est d'apprendre un réseau à reconnaître des images de bananes, tomates, pizza et sushis.


```python
import numpy as np
import matplotlib.pyplot as plt
import os
from tempfile import TemporaryDirectory
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
```

On utilise une technique d'[augmentation de données](https://pytorch.org/vision/stable/transforms.html) pour augmenter la taille de la base d'entraînement. On normalise les images d'entraînement et de validation.

```python
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = './data'
batch_size = 4
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=6) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = "cuda" if torch.cuda.is_available() else "cpu"
```

On donne une fonction d'affichage d'images ({numref}`sushis`), renormalisées.

```python
def imshow(I, title=None):
    I = I.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    I = mean + std * I 
    I = np.clip(I, 0, 1)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(I)
    plt.savefig('./images.png',dpi=100)
    if title is not None:
        plt.title(title)

inputs, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])
```

```{figure} ./images/sushis.png
:name: sushis
Quelques exemples d'images.
```


Et une fonction d'affichage des prédictions des modèles sur l'ensemble de validation

```python
def predict(model):
    was_training = model.training
    model.eval()

    with torch.no_grad():
        # Extraction d'un batch d'évaluation
        inputs, labels = next(iter(dataloaders['val']))
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        out = torchvision.utils.make_grid(inputs)
        imshow(out, title=[class_names[preds[j]] for j in range(inputs.size()[0])])

        model.train(mode=was_training)


```

### Premier entraînement

Dans ce premier entraînement, on utilise un réseau pré-entraîné, qui sert d'initialisation à un entraînement complet sur la base d'entraînement.

```python
def train1(model, criterion, optimizer, scheduler, num_epochs=25):

    # Répertoire temporaire pour les checkpoints 
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_params.pt')
        torch.save(model.state_dict(), best_model_params_path)

        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('*' * 10)

            # Le modèle est utilisé en entraînement et en évaluation
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  
                else:
                    model.eval()   

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    # passe avant 
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # rétropropagation et mise à jour des poids en entraînement 
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                # Mise à jour du learning rate
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Perte: {epoch_loss:.4f} Précision: {epoch_acc:.4f}')

                # Sauvegarde du modèle si meilleur
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)
                    
        # Chargement du meilleur modèle
        model.load_state_dict(torch.load(best_model_params_path))
    return model

```

Le réseau utilisé est ResNet34, entraîné sur ImageNet. On ajoute une couche de classification spécifique au problème et on entraîne le tout.

```python
model1 = models.resnet34(weights='IMAGENET1K_V1')
# Nombre de caractéristiques extraites avant le réseau de classification
nb = model1.fc.in_features

# Ajout d'une couche de classification spécifique
model1.fc = nn.Linear(nb, len(class_names))
model1 = model1.to(device)

# Fonction de perte
criterion = nn.CrossEntropyLoss()

# Tous les poids vont être optimisés, y compris ceux du réseau convolutif.
optimizer = optim.SGD(model1.parameters(), lr=0.001, momentum=0.9)
lr_sch = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model1 = train_model(model1, criterion, optimizer,lr_sch, num_epochs=25)

predict(model1)
```

```{figure} ./images/val.png
:name: val
Quelques exemples d'images de validation étiquetées.
```


### Second entraînement

Ici, on ne réentraîne pas les poids du réseau convolutif. On laisse donc ce réseau agir comme un extracteur de caractéristiques et on entraîne uniquement les poids de la couche de classification ajouté en bout.

```python
model2 = torchvision.models.resnet34(weights='IMAGENET1K_V1')
# On fige les poids du réseau convolutif
for param in model2.parameters():
    param.requires_grad = False

# On ajoute une couche de classification
nb = model2.fc.in_features
model2.fc = nn.Linear(nb, len(class_names))

model2 = model2.to(device)

criterion = nn.CrossEntropyLoss()

# On optimise juste les poids de la couche de classification
optimizer = optim.SGD(model2.fc.parameters(), lr=0.001, momentum=0.9)
lr_sch = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model2 = train_model(model2, criterion, optimizer,lr_sch, num_epochs=25)

predict(model2)

```



```{bibliography}
:style: unsrt
```

