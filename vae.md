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

::: frame
::: center
![image](../../images/VAE2){height="80pt"}
:::
:::

# Generative models

::: frame
Generative models

::: Lblock
Learning a model that represents the distribution of data, with input
training samples

$$\mathcal{P}_{model}(x)\sim  \mathcal{P}_{data}(x)$$
:::

![image](../../images/mnist){width=".4\\textwidth"}$\quad \Longrightarrow\quad$
![image](../../images/mnist2){width=".3\\textwidth"}
:::

::: frame
Generative models

![image](../../images/generativemodels){width="\\textwidth"}\
Source: G Louppe
:::

::: frame
Generative models

### The landscape of generative models

![image](../../images/landscapegen){width="\\textwidth"}\
Source: Song et al., CVPR 2023.

::: Lblock
-   Variational Autoencoders

-   Generative Adversarial Networks
:::
:::

# Latent models

::: frame
Latent models A latent variable model relates a set of observable
variables $\boldsymbol x\in X$ to a set of latent variables
$\boldsymbol h\in H$
$$\textbf{\textsf{\textup{p}}}(\boldsymbol x,\boldsymbol h) = \textbf{\textsf{\textup{p}}}(\boldsymbol x|\boldsymbol h)\textbf{\textsf{\textup{p}}}(\boldsymbol h)$$
if $\boldsymbol h$ are causal factors for $\boldsymbol x$ $\Rightarrow$
sampling from
$\textbf{\textsf{\textup{p}}}(\boldsymbol x|\boldsymbol h)$ = generative
process from $H$ to $X$.

::: Lblock
Inference: given
$\textbf{\textsf{\textup{p}}}(\boldsymbol x,\boldsymbol h)$, compute
$$\textbf{\textsf{\textup{p}}}(\boldsymbol h|\boldsymbol x) = \frac{\textbf{\textsf{\textup{p}}}(\boldsymbol x|\boldsymbol h)\textbf{\textsf{\textup{p}}}(\boldsymbol h)}{\textcolor{red}{\textbf{\textsf{\textup{p}}}(\boldsymbol x)}}$$
[Intractable]{style="color: red"}
:::
:::

::: frame
Latent models

::: Lblock
-   $\textbf{\textsf{\textup{p}}}_{\phi} (\boldsymbol{h}|\boldsymbol x)$:
    family of distributions approximating
    $\textbf{\textsf{\textup{p}}}(\boldsymbol{h}|\boldsymbol x)$

-   $\phi$ is optimized to minimize the \"distance\" between both
    distributions.

-   Among all similarity measures: Kullback Leibler divergence

$$\begin{aligned}
KL( \textbf{\textsf{\textup{p}}}_{\phi} (\boldsymbol{h}|\boldsymbol x)|| \textbf{\textsf{\textup{p}}}(\boldsymbol h| \boldsymbol  x)) &=& \mathbb{E}_{ \textbf{\textsf{\textup{p}}}_{\phi} (\boldsymbol{h}|\boldsymbol x)} \left [log\frac{ \textbf{\textsf{\textup{p}}}_{\phi} (\boldsymbol{h}|\boldsymbol x)}{ \textbf{\textsf{\textup{p}}}(\boldsymbol{h}|\boldsymbol x)} \right ]\\
&=& \mathbb{E}_{ \textbf{\textsf{\textup{p}}}_{\phi} (\boldsymbol{h}|\boldsymbol x)}  \left [ log(\textbf{\textsf{\textup{p}}}_{\phi} (\boldsymbol{h}|\boldsymbol x))-log(\textbf{\textsf{\textup{p}}}(\boldsymbol{x},\boldsymbol h))\right ] +\textcolor{red}{log(\textbf{\textsf{\textup{p}}}(\boldsymbol x))}
\end{aligned}$$ [Still intractable]{style="color: red"}
:::
:::

::: frame
Latent models But\... $$\begin{aligned}
\displaystyle\min_\phi KL( \textbf{\textsf{\textup{p}}}_{\phi} (\boldsymbol{h}|\boldsymbol x)|| \textbf{\textsf{\textup{p}}}(\boldsymbol h| \boldsymbol  x)) 
&=&\displaystyle\min_\phi log(\textbf{\textsf{\textup{p}}}(\boldsymbol x))- \mathbb{E}_{ \textbf{\textsf{\textup{p}}}_{\phi} (\boldsymbol{h}|\boldsymbol x)}  \left [log(\textbf{\textsf{\textup{p}}}(\boldsymbol{x},\boldsymbol h)) - log (\textbf{\textsf{\textup{p}}}_{\phi} (\boldsymbol{h}|\boldsymbol x))\right ]\\
&=&\displaystyle\max_\phi \underbrace{\mathbb{E}_{ \textbf{\textsf{\textup{p}}}_{\phi} (\boldsymbol{h}|\boldsymbol x)}  \left [log(\textbf{\textsf{\textup{p}}}(\boldsymbol{x},\boldsymbol h)) - log (\textbf{\textsf{\textup{p}}}_{\phi} (\boldsymbol{h}|\boldsymbol x))\right ]}_{ELBO(\boldsymbol x,\phi)}
\end{aligned}$$
:::

# VAE

::: frame
Autoencoders

myTrapezium/.pic = (0,0) -- (0,)̱ -- (,)̧ -- (,-)̧ -- (0,-)̱ -- cycle ;
(-center) at (/2,0); (-out) at (,0);

= \[thick, decoration=markings,mark=at position 1 with , double
distance=1.4pt, shorten \>= 5.5pt, preaction = decorate, postaction =
draw,line width=1.4pt, white,shorten \>= 4.5pt\]

::: center
:::

::: Lblock
-   A neural network trained using unsupervised learning

-   Trained to copy its input to its output

-   Learns an embedding $h$

$$\hat{\boldsymbol x} = g[f(\boldsymbol x)]\quad h = f(\boldsymbol x)$$
:::
:::

::: frame
Variational Autoencoders

::: center
![image](../../images/xhx){width=".8\\textwidth"}
:::
:::

::: frame
ELBO

::: Lblock
$$\displaystyle\min_{\phi,\theta} (-ELBO(\boldsymbol x,\phi)) = -\mathbb{E}_{ \textbf{\textsf{\textup{p}}}_{\phi} (\boldsymbol{h}|\boldsymbol x)}  \left [\textcolor{bluelimos}{log(\textbf{\textsf{\textup{p}}}_\theta(\boldsymbol{x}|\boldsymbol h))}\right ]+\textcolor{orange}{KL( \textbf{\textsf{\textup{p}}}_{\phi} (\boldsymbol{h}|\boldsymbol x)|| \textbf{\textsf{\textup{p}}}(\boldsymbol h))}$$

-   Given $\theta$, optimize $\phi$ so that latent variable distribution
    explains the observed data, while remaining close to the prior

-   Given $\phi$, optimize $\theta$ so that observed data is well
    explained by the latent variables.
:::
:::

::: frame
Variational Autoencoders
:::

::: frame
Variational Autoencoders

Let $\boldsymbol x_1\cdots \boldsymbol x_n$ be the training inputs.
$\textcolor{orange}{KL( \textbf{\textsf{\textup{p}}}_{\phi} (\boldsymbol{h}|\boldsymbol x)|| \textbf{\textsf{\textup{p}}}(\boldsymbol h))} =\frac 1n\left [ -\frac 12 \displaystyle\sum_{i=1}^d \left ( 1+log\sigma^f_i(\boldsymbol x)-(\mu^f_i(\boldsymbol x))^2-\sigma^f_i(x)\right )\right ]$
:::

::: frame
Reparameterization trick

::: columns
::: Lblock
Expressing $\boldsymbol h$ as some differentiable and invertible
transformation of another random variable $\epsilon$ given
$\boldsymbol x$ and $\phi$.

$$\boldsymbol h = \mu(\boldsymbol x,\phi) + \sigma(\boldsymbol x,\phi)\odot \epsilon,\quad \epsilon\sim \mathcal{N}(0,I)$$
:::
:::

Gradient can flow out of any random variable $\Rightarrow$
backpropagation is possible.
:::

::: frame
Summary ![image](../../images/VAE2){height="80pt"}

::: Lblock
1.  Define Encoder and Decoder

2.  Define the latent space distribution using the reparametrization
    trick

3.  Define the ELBO loss

4.  Train and play with VAE !
:::
:::

# Latent space

::: frame
Exploration

-   Changing one single variable $h_i$ and keep all other $h_j$ fixed.

-   Dimensions of $\boldsymbol h$ encode different interpretable latent
    features.

::: center
![image](../../images/mnistlatent){width=".45\\textwidth"}
![image](../../images/fashionmnistlatent){width=".45\\textwidth"}
:::
:::

::: frame
Disentanglement
![image](../../images/disentangle){width=".7\\textwidth"}\
Kingma et al, 2014
:::

::: frame
Disentanglement
![image](../../images/disentangle2){width="\\textwidth"} Amini and
Soleimany, 2019
:::

::: frame
Disentanglement $\beta-VAE$: enforces disentanglement
$$\displaystyle\min_{\phi,\theta} (-ELBO(\boldsymbol x,\phi)) = -\mathbb{E}_{ \textbf{\textsf{\textup{p}}}_{\phi} (\boldsymbol{h}|\boldsymbol x)}  \left [\textcolor{bluelimos}{log(\textbf{\textsf{\textup{p}}}_\theta(\boldsymbol{x}|\boldsymbol h))}\right ]+\beta \textcolor{orange}{KL( \textbf{\textsf{\textup{p}}}_{\phi} (\boldsymbol{h}|\boldsymbol x)|| \textbf{\textsf{\textup{p}}}(\boldsymbol h))}$$
![image](../../images/betavae){width=".6\\textwidth"} Higgins et al.,
2017
:::

::: frame
Exploration

::: center
![image](../../images/interpol){width=".7\\textwidth"}
:::

::: columns
\
Interpolation in the image space

\
Interpolation in the latent space
:::
:::

::: frame
Sampling ![image](../../images/vaesampling){width=".8\\textwidth"}
:::

# Implementation

::: frame
Encoder-Decoder myTrapezium/.pic = (0,0) -- (0,)̱ -- (,)̧ -- (,-)̧ -- (0,-)̱
-- cycle ; (-center) at (/2,0); (-out) at (,0);

= \[thick, decoration=markings,mark=at position 1 with , double
distance=1.4pt, shorten \>= 5.5pt, preaction = decorate, postaction =
draw,line width=1.4pt, white,shorten \>= 4.5pt\]

::: center
:::

::: Lblock
-   MLP

-   CNN

-   RNN
:::
:::

::: frame
Reparameterization trick myTrapezium/.pic = (0,0) -- (0,)̱ -- (,)̧ -- (,-)̧
-- (0,-)̱ -- cycle ; (-center) at (/2,0); (-out) at (,0);

= \[thick, decoration=markings,mark=at position 1 with , double
distance=1.4pt, shorten \>= 5.5pt, preaction = decorate, postaction =
draw,line width=1.4pt, white,shorten \>= 4.5pt\]

::: center
:::

$$\boldsymbol h = \mu(\boldsymbol x,\phi) + \sigma(\boldsymbol x,\phi)\odot \epsilon,\quad \epsilon\sim \mathcal{N}(0,I)$$

::: center
![image](../../images/trick){width="\\textwidth"}
:::
:::

::: frame
KL divergence
$$\textcolor{orange}{KL( \textbf{\textsf{\textup{p}}}_{\phi} (\boldsymbol{h}|\boldsymbol x)|| \textbf{\textsf{\textup{p}}}(\boldsymbol h))} =\frac 1n\left [ -\frac 12 \displaystyle\sum_{i=1}^d \left ( 1+log\sigma^f_i(\boldsymbol x)-(\mu^f_i(\boldsymbol x))^2-\sigma^f_i(x)\right )\right ]$$

::: center
![image](../../images/KL){width="\\textwidth"}
:::
:::

::: frame
And now gather all the stuff

::: center
![image](../../images/allvae){width="\\textwidth"}
:::
:::
