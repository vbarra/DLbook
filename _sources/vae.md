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
# Autoencodeurs variationnels
Les autoencodeurs variationnels sont, comme les GAN (abordés dans le cours suivant), des modèles génératifs, c'est-à-dire des modèles probabilistes $p$ pouvant être utilisés pour 
simuler (ou générer) des données réalistes $\boldsymbol x\sim p(\boldsymbol x,\boldsymbol\theta)$, aussi proches que possible de la vraie (mais inconnue) distribution des données $p(\boldsymbol x)$, pour laquelle seul un échantillon de données est disponible.

Le paysage de ces modèles génératifs s'est beaucoup peuplé depuis 2014 ({numref}`landscape`).



```{figure} ./images/landscapegen.png
:name: landscape
Paysage des modèles génératifs (Source: [Song et al., CVPR 2023](https://cvpr2023.thecvf.com/virtual/2023/tutorial/18546))
```

Dans ce cours, nous aborderons uniquement les autoencodeurs variationnels (VAE) et les réseaux antagonistes générateurs (GAN).



## Inférence variationnelle

### Modèles à variables latentes

Un modèle à variables latentes met en relation un ensemble de variables observables $\boldsymbol x\in \mathcal X$ avec un ensemble de variables latentes  $\boldsymbol h\in \mathcal H$

$$p(\boldsymbol x,\boldsymbol h) = p(\boldsymbol x|\boldsymbol h)p(\boldsymbol h)$$

Si $\boldsymbol h$ sont des facteurs causaux pour  $\boldsymbol x$, alors échantillonner selon $p(\boldsymbol x|\boldsymbol h)$ permet de créer un modèle génératif de  $\mathcal H$ vers $\mathcal X$.

Pour l'inférence, étant donnée $p(\boldsymbol x,\boldsymbol h)$, il "suffit" de calculer 

$$ p(\boldsymbol h|\boldsymbol x) = \frac{p(\boldsymbol x|\boldsymbol h)p(\boldsymbol h)}{p(\boldsymbol x)}$$

Malheureusement, $p(\boldsymbol x)$ est inaccessible.

### Inférence variationnelle

L'inférence variationnelle transforme l'estimation de $p(\boldsymbol h|\boldsymbol x)$ en un problème d'optimisation.

On considère une famille de distributions $q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)$ approchant $p(\boldsymbol{h}|\boldsymbol x)$, où $\boldsymbol \phi$ sont les paramètres variationnels. Ces paramètres sont optimisés pour minimiser une distance entre $q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)$ et $p(\boldsymbol x,\boldsymbol h)$. Parmi toutes les distances possibles, on retient la divergence de Kullback-Leibler

 $$KL(p\|q) = \sum_{x \in X} p(x) \log \frac{p(x)}{q(x)}$$

et ainsi : 

$$\begin{align}
KL( q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)|| p(\boldsymbol h| \boldsymbol  x)) &=& \mathbb{E}_{ q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)} \left [log\frac{ q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)}{ p(\boldsymbol{h}|\boldsymbol x)} \right ]\\
&=& \mathbb{E}_{ q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)}  \left [ log(q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x))-log(p(\boldsymbol{x},\boldsymbol h))\right ] +log(p(\boldsymbol x))
\end{align}$$

Le dernier terme $log(p(\boldsymbol x)$ reste cependant toujours inaccessible.

Cependant, 

$$\begin{align}
\displaystyle\min_\boldsymbol\phi KL( q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)|| p(\boldsymbol h| \boldsymbol  x)) 
&=&\displaystyle\min_\phi log(p(\boldsymbol x))- \mathbb{E}_{ q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)}  \left [log(p(\boldsymbol{x},\boldsymbol h)) - log (q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x))\right ]\\
&=&\displaystyle\max_\phi \underbrace{\mathbb{E}_{ q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)}  \left [log(p(\boldsymbol{x},\boldsymbol h)) - log (q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x))\right ]}_{ELBO(\boldsymbol x,\phi)}
\end{align}$$

avec ELBO(Evidence Lower Bound Objective) définie par 

$$\begin{align}
ELBO(\boldsymbol x,\boldsymbol\phi)&=&\mathbb{E}_{ q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)}  \left [log(p(\boldsymbol{x},\boldsymbol h)) - log (q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x))\right ] \\
&=& \mathbb{E}_{ q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)}  \left [log(p(\boldsymbol{x}|\boldsymbol h))p(\boldsymbol h) - log (q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x))\right ]\\
&=& \mathbb{E}_{ q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)}  \left [\color{blue}{log(p(\boldsymbol{x}|\boldsymbol h))}\right ]-\color{red}{KL( q_{\boldsymbol\phi} (\boldsymbol{h}|\boldsymbol x)|| p(\boldsymbol h))}
\end{align}$$

Dans l'équation prédécente, on a éliminé $log(p(\boldsymbol x))$ qui ne dépend pas de $\boldsymbol\phi$.

En maximisant la fonction $ELBO(\boldsymbol x,\boldsymbol\phi)$ :
- <span style="color:blue"> le premier terme encourage les distributions à converger dans les configurations des variables latentes $\boldsymbol h$ expliquant les données observées </span>
- <span style="color:red"> le second terme force les distribution à être proches du prior.</span>

Finalement, étant donné un échantillon  $E_a = \{\boldsymbol x_i,i\in[\![1,N]\!]\}$, la fonction objectif finale est 

$$\displaystyle\sum_{\boldsymbol x_i\in E_a}ELBO(\boldsymbol x_i,\boldsymbol \phi)$$

Pour la maximiser, on peut utiliser une montée de gradient, maix $\nabla_{\boldsymbol\phi}ELBO(\boldsymbol x_i,\boldsymbol \phi)$ est en général difficile à calculer.


## Autoencodeurs variationnels

### Principe

Un autoencodeur variationnel (VAE) est un modèle profond à variables latentes ({numref}`vae0`) tel que :
- $p(\boldsymbol h)$ est prescrit à l'avance
- la vraisemblance $p_{\boldsymbol\theta}(\boldsymbol x|\boldsymbol h)$ est un décodeur (réseau génératif) $D_{\boldsymbol\theta}$ tel que $\boldsymbol\Phi = D_{\boldsymbol\theta}(\boldsymbol h)$ où $\boldsymbol \Phi$  sont les paramètres de la distribution des données. Par exemple 

$$\mu,\sigma  = D_{\boldsymbol\theta}(\boldsymbol h), \quad p_{\boldsymbol\theta}(\boldsymbol x|\boldsymbol h) = \mathcal N(\boldsymbol x,\mu,\sigma^2 \boldsymbol I)$$
- la distribution approchée $q_{\boldsymbol\phi}(\boldsymbol h|\boldsymbol x)$ est paramétrée par un encodeur (réseau d'inférence) $E_{\boldsymbol\phi}$  tel que $\boldsymbol\nu = E_{\boldsymbol\phi}(\boldsymbol x)$ sont les paramètres de la distribution approchée. Par exemple : 

$$ \mu,\sigma  = \boldsymbol\nu = E_{\boldsymbol\phi}(\boldsymbol x)\quad q_{\boldsymbol\phi}(\boldsymbol h|\boldsymbol x) = \mathcal N(\boldsymbol h,\mu,\sigma^2 \boldsymbol I)$$



```{figure} ./images/vae0.png
:name: vae0
Architecture générale d'un autoencodeur variationnel
```

Comme précédemment, on peut utiliser l'inférence variationnelle pour optimiser de manière jointe $\boldsymbol\theta$ et $\boldsymbol\phi$ : 

$$
\begin{align}
\boldsymbol\theta^*,\boldsymbol\phi^* &=& Arg\; max_{\boldsymbol\theta,\boldsymbol\phi}\; ELBO(\boldsymbol x,\boldsymbol\theta,\boldsymbol\phi)\\
&=&Arg\; max_{\boldsymbol\theta,\boldsymbol\phi}\left [\mathbb{E}_{q_{\boldsymbol\phi}(\boldsymbol h|\boldsymbol x)}\left (log(p_{\boldsymbol\theta}(\boldsymbol x|\boldsymbol h))\right )-KL(q_{\boldsymbol\phi}(\boldsymbol h|\boldsymbol x)||p(\boldsymbol h))\right ]
\end{align}
$$

Etant donné $D_{\boldsymbol\theta}$, on veut ajuster les variables latentes, en optimissant $\boldsymbol\phi$, de sorte à ce qu'elles expliquent les données observées, tout en restant près de données générées par $p(\boldsymbol h)$. De même, étant donné $E_{\boldsymbol\phi}$, on veut ajuster les variables observées, en optimissant $\boldsymbol\theta$, de telle sorte qu'elles soient le plus possible expliquées par les variables latentes.

L'optimisation peut se faire par montée de gradients : 
- $\nabla_{\boldsymbol\theta}ELBO(\boldsymbol x,\boldsymbol\theta,\boldsymbol\phi) = \mathbb{E}_{q_{\boldsymbol\phi}(\boldsymbol h|\boldsymbol x)}\left [\nabla_{\boldsymbol\theta}(log p_{\boldsymbol\theta}(\boldsymbol x|\boldsymbol h))\right ]$ peut être estimé (par exemple par méthode de Monte Carlo)
- $\nabla_{\boldsymbol\phi}ELBO(\boldsymbol x,\boldsymbol\theta,\boldsymbol\phi)$ est plus difficile à estimer (on ne peut rétropropager le gradient à travers $\boldsymbol h$ pour calculer $\nabla_{\boldsymbol\phi}$)

La solution à ce problème est appelée *astuce de reparamétrisation* (reparameterization trick) : on exprime $\boldsymbol h$ à l'aide d'une transformation différentiable et inversible $F$ d'une autre variable aléatoire $\varepsilon$, étant donnés $\boldsymbol x$ et $\boldsymbol\phi$, de telle sorte que la distribution de $\varepsilon$ est indépendante de  $\boldsymbol x$ et $\boldsymbol\phi$ : 

$$\boldsymbol h = F(\boldsymbol\phi,\boldsymbol x,\boldsymbol\varepsilon)$$

Un choix classique est 

$$\boldsymbol h = \mu(\boldsymbol x,\phi) + \sigma(\boldsymbol x,\phi)\odot \varepsilon,\quad \varepsilon\sim \mathcal{N}(0,I)$$ 

On montre alors que $\nabla_{\boldsymbol\phi}ELBO(\boldsymbol x,\boldsymbol\theta,\boldsymbol\phi)$ peut être estimé par méthode de Monte Carlo.

### Exemple
Soit $\boldsymbol h\in\mathbb{R}^d$. On suppose que la distribution de la variable latente suit une loi normale centrée réduite $p(\boldsymbol h) = \mathcal N(\boldsymbol 0,\boldsymbol I)$.

Alors 

$$p_{\boldsymbol\theta}(\boldsymbol x|\boldsymbol h) = \mathcal N(\boldsymbol x,\boldsymbol\mu_{\boldsymbol\theta}(\boldsymbol h),\sigma^2_{\boldsymbol\theta}(\boldsymbol h)\boldsymbol I)$$

où on modélise le décodeur $D_{\boldsymbol\theta}$ par : 
- $\boldsymbol\mu_{\boldsymbol\theta}(\boldsymbol h) = \boldsymbol W_1^\top \boldsymbol z + \boldsymbol b_1$
- $log(\boldsymbol\sigma^2_{\boldsymbol\theta}(\boldsymbol h)) = \boldsymbol W_2^\top \boldsymbol z + \boldsymbol b_2$
- $\boldsymbol z = ReLU(\boldsymbol W_3^\top \boldsymbol h + \boldsymbol b_3)$

avec donc $\boldsymbol\theta = (\boldsymbol W_1,\boldsymbol W_2,\boldsymbol W_3,\boldsymbol b_1,\boldsymbol b_2,\boldsymbol b_3)$

De même, on modélise l'encodeur $E_{\boldsymbol\phi}$ par : 

$$q_{\boldsymbol\phi}(\boldsymbol h|\boldsymbol x) = \mathcal N(\boldsymbol h,\boldsymbol\mu_{\boldsymbol\phi}(\boldsymbol x),\sigma^2_{\boldsymbol\phi}(\boldsymbol x)\boldsymbol I)$$

où 
- $p(\varepsilon) = \mathcal N(\boldsymbol 0,\boldsymbol I)$
- $\boldsymbol h = \boldsymbol\mu_{\boldsymbol\phi}(\boldsymbol x) + \sigma_{\boldsymbol\phi}(\boldsymbol x)\odot\varepsilon$
- $\boldsymbol\mu_{\boldsymbol\phi}(\boldsymbol x) = \boldsymbol W_4^\top \boldsymbol z + \boldsymbol b_4$
- $log(\boldsymbol\sigma^2_{\boldsymbol\phi}(\boldsymbol x)) = \boldsymbol W_5^\top \boldsymbol z + \boldsymbol b_5$
- $\boldsymbol z = ReLU(\boldsymbol W_6^\top \boldsymbol x + \boldsymbol b_6)$

avec donc $\boldsymbol\phi = (\boldsymbol W_4,\boldsymbol W_5,\boldsymbol W_6,\boldsymbol b_4,\boldsymbol b_5,\boldsymbol b_6)$


Le calcul de ELBO est alors 

$$

\begin{align}
ELBO(\boldsymbol x,\boldsymbol\theta,\boldsymbol\phi) &=& \mathbb{E}_{q_{\boldsymbol\phi}(\boldsymbol h|\boldsymbol x)}\left (log(p_{\boldsymbol\theta}(\boldsymbol x|\boldsymbol h))\right )-KL(q_{\boldsymbol\phi}(\boldsymbol h|\boldsymbol x)||p(\boldsymbol h))\\
&=& \mathbb{E}_{p(\boldsymbol\varepsilon)}\left (log(p_{\boldsymbol\theta}(\boldsymbol x|\boldsymbol h=F(\boldsymbol\phi,\boldsymbol x,\boldsymbol\varepsilon)))\right )-KL(q_{\boldsymbol\phi}(\boldsymbol h|\boldsymbol x)||p(\boldsymbol h))
\end{align}
$$

où 

$$KL(q_{\boldsymbol\phi}(\boldsymbol h|\boldsymbol x)||p(\boldsymbol h)) = \frac12\displaystyle\sum_{i=1}^d \left (1+log({\sigma_j}_{\boldsymbol\phi}^2(\boldsymbol x)) - {\mu_j}^2_{\boldsymbol\phi}(\boldsymbol x)-{\sigma_j}_{\boldsymbol\phi}^2(\boldsymbol x)\right )$$

dont les dérivées peuvent être évaluées analytiquement.

### En résumé

Pour utiliser un VAE il suffit donc 
- de définir l'encodeur et le décodeur
- de définir la distribution de l'espace latent en utilisant l'astuce de reparamétrisation
- de définir la fonction de perte ELBO
- d'entraîner le tout et d'apprécier les données générées !

## Implémentation

On propose ici d'implémenter un auto-encodeur variationnel, et une variation de ce modèle intégrant une génération conditionnelle. Le jeu de données utilisé est [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist).

```python
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
```

On se donne des fonctions d'affichage des images reconstruites et générées.

```python
def show(img):
    plt.imshow(np.transpose(img.numpy(), (1,2,0)), interpolation='nearest')
    
# Affichage d'images reconstruites
def plot_reconstruction(model, n=24):
    x,_ = next(iter(data_loader))
    x = x[:n,:,:,:].to(device)
    try:
        out, _, _, log_p = model(x.view(-1, image_size)) 
    except:
        out, _, _ = model(x.view(-1, image_size)) 
    x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
    out_grid = torchvision.utils.make_grid(x_concat).cpu().data
    show(out_grid)

# Affichage d'images générées par le VAE classique
def plot_generation(model, n=24):
    with torch.no_grad():
        z = torch.randn(n, z_dim).to(device)
        out = model.decode(z).view(-1, 1, 28, 28)

    out_grid = torchvision.utils.make_grid(out).cpu()
    show(out_grid)

# Affichage d'images générées par le VAE où la génération est conditionnée par la classe.
def plot_conditional_generation(model, n=8):
    plt.figure()
    with torch.no_grad():
        matrix = np.zeros((n,n_classes))
        matrix[:,0] = 1

        final = matrix[:]
        for i in range(1,n_classes):
            final = np.vstack((final,np.roll(matrix,i)))
        z = torch.randn(8, z_dim)
        z = z.repeat(n_classes,1).to(device)
        y_onehot = torch.tensor(final).type(torch.FloatTensor).to(device)
        out = model.decode(z,y_onehot).view(-1, 1, 28, 28)

    out_grid = torchvision.utils.make_grid(out).cpu()
    show(out_grid)
```

On charge ensuite les données

```python
data_dir = 'data'
trainbatch_size = 128
testbatch_size = 16

dataset = torchvision.datasets.FashionMNIST(root=data_dir,train=True,transform=transforms.ToTensor(),download=True)
data_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=trainbatch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor()),batch_size=testbatch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### VAE classique

On construit tout d'abord un VAE classique

```python
image_size = 784
h_dim = 400
z_dim = 20
num_epochs = 15
learning_rate = 1e-3
```

```python
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
    
    # Astuce de reparamétrisation
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```

On entraîne ensuite ce modèle
```python 
Recon = []
KL = []

for epoch in range(num_epochs):
    for i,(x, _) in enumerate(data_loader):
        x = x.to(device).view(-1, image_size)
        x_reconst, mu, log_var = model(x)
        
        reconst_loss = F.mse_loss(x_reconst, x, reduction='sum')
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        loss = reconst_loss + kl_div
        if i%10==0:
            Recon.append( reconst_loss.item()/len(x))
            KL.append(kl_div.item()/len(x))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
plt.plot(Recon,color='r',label='Reconstruction Loss')
plt.plot(KL,color='b',label='Divergence KL')
plt.legend(loc='best')
```

```{figure} ./images/recon.png
:name: recon
Apprentissage du VAE
```

On affiche alors des exemples d'images reconstruites
```python
plot_reconstruction(model)
```

```{figure} ./images/reconstruction.png
:name: recons
Exemples d'images reconstruites $\hat x = D(E(x))$ 
```


et des exemples d'images générées
```python
plot_generation(model)
```

```{figure} ./images/generation.png
:name: generation
Exemples d'images générées $x=D(h)$
```

