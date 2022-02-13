#!/usr/bin/env python
# coding: utf-8

# # Coin toss problem
# 
# ## What is the problem?
# 
# Let's pick up a random coin (not necessarily a fair one with equal probability of head and tail). We did the coin toss experiment $n$ times and gathered the observed data $D$ as a set of outcomes (e.g. $\{H, T, T, ...\}$). Now, we are interested in predicting the probability of heads $p(H)=\theta_{best}$ for our coin.
# 
# ## Applying Bayes rule
# 
# In this problem, we will {ref}`model the distribution of parameters <parameters-framework>`. 
# 
# \begin{equation}
# \underbrace{p(\theta|D)}_{\text{Posterior}} = \frac{\overbrace{p(D|\theta)}^{\text{Likelihood}}}{\underbrace{p(D)}_{\text{Evidence}}}\underbrace{p(\theta)}_{\text{Prior}}
# \end{equation}
# \begin{equation}
# p(D) = \int_{\theta}p(D|\theta)p(\theta)d\theta
# \end{equation}
# 
# We are interested in $p(\theta|D)$ and to derive that, we need prior, likelihood and evidence terms. Let us look at them one by one.
# 
# ### Prior
# 
# What is our prior belief about the coin's probability of head $p($H$)$? Yes, that's exactly the question. A most simple way is to assume equal probability of heads and tails. However, we can represent our prior belief in terms of a distribution. Let's assume a beta distribution over the probability of heads $p(H) = \theta$ (we will see in later sections why beta and not Gaussian or uniform or something else?). So, our prior distibution $p(\theta)$ is:
# 
# $$
# p(\theta|\alpha, \beta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)}, \alpha,\beta>0\\
# B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}\\
# \Gamma(\alpha) = (\alpha-1)!
# $$
# 
# Here, $\alpha$ and $\beta$ are the hyperparameters of the beta distrubution. $B$ is Beta function. You may play with [this interactive demo](https://huggingface.co/spaces/Zeel/Beta_distribution) to see how pdf changes with $\alpha$ and $\beta$. In our modeling, we can assume that $\alpha$ and $\beta$ are already known. There are methods of assuming distributions over the $\alpha$ and $\beta$ as well but that's out of the scope for now.
# 
# ### Likelihood
# 
# Likelihood is probability of observing the data $D$ given $\theta$. From, $n$ number of experiments, if we received heads $h$ times, then $p(D|\theta)$ follows a Bernoulli distribution. We can also arrive at this formula by following the basic probability rules for independent events:
# 
# $$
# p(D|\theta) = \theta^h(1-\theta)^{n-h}
# $$
# 
# ### Maximum likelihood estimation (MLE)
# 
# In cases, where prior is not available, we can use likelihood to get the best estimate of $\theta$. Let us find the optimal theta by differentiating likelihood $p(D|\theta)$ w.r.t $\theta$.
# 
# \begin{align}
# p(D|\theta) &= (\theta)^h(1-\theta)^{n-h}\\
# \text{taking log both sides to simplify things,}\\
# \log p(D|\theta) &= h\log(\theta)+(n-h)\log(1-\theta)\\
# \frac{d}{d\theta}\log p(D|\theta) &= \frac{h}{\theta} - \frac{n-h}{1-\theta} = 0\\
# &= h(1-\theta)-(n-h)\theta = 0\\
# &= h - h\theta - n\theta + h\theta = 0\\
# \therefore \theta_{MLE} = \frac{h}{n}
# \end{align}
# 
# How can we know if optima at $\theta_{MLE}$ is a maxima? well, it is a maxima if $\frac{d^2}{d\theta^2}\log p(D|\theta)$ is negative [(check here if not convinced)](https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/optimizing-multivariable-functions/a/second-partial-derivative-test):
# 
# \begin{align}
# \frac{d}{d\theta}\log p(D|\theta) &= \frac{h}{\theta} - \frac{n-h}{1-\theta}\\
# \frac{d^2}{d\theta^2}\log p(D|\theta) &= -\frac{h}{\theta^2}-\frac{n-h}{(1-\theta)^2}
# \end{align}
# 
# After a bit of thinking, one can see that above value is always negative and thus our optima is a maxima.
# 
# ### Maximum a posteriori estimation (MAP)
# 
# We know that posterior is given by the following formula:
# 
# \begin{equation}
# \underbrace{p(\theta|D)}_{\text{Posterior}} = \frac{\overbrace{p(D|\theta)}^{\text{Likelihood}}}{\underbrace{p(D)}_{\text{Evidence}}}\underbrace{p(\theta)}_{\text{Prior}}
# \end{equation}
# 
# If we are only interested in maximum probable value of $\theta$ in the posterior (point estimate in other words), we can differentiate the posterior w.r.t. $\theta$. However, we have not yet derived the evidence but it does not depend on $\theta$. So, we can claim that the following is true:
# 
# $$
# \arg \max_{\theta} p(\theta|D) = \arg \max_{\theta} p(D|\theta)p(\theta)
# $$
# 
# Now, differentiating $p(D|\theta)p(\theta)$ w.r.t $\theta$:
# 
# \begin{align}
# p(D|\theta)p(\theta) &= \theta^h(1-\theta)^{N-h}\cdot\frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)}\\
#                      &= \frac{\theta^{h+\alpha-1}(1-\theta)^{N-h+\beta-1}}{B(\alpha, \beta)}\\
# \text{Taking log for simplification}\\
# \log p(\theta|D)p(\theta) &= (h+\alpha-1)\log(\theta) + (N-h+\beta-1)\log(1-\theta) - \log(B(\alpha, \beta))\\
# \\
# \frac{d}{d\theta} \log p(\theta|D)p(\theta) &= \frac{h+\alpha-1}{\theta} - \frac{N-h+\beta-1}{1-\theta} = 0\\
# \\
# \therefore \theta_{MAP} = \frac{h+(\alpha-1)}{N+(\alpha-1)+(\beta-1)}
# \end{align}
# 
# Now, we have the maximum probable value of $\theta$ from the posterior but if we are interested in the posterior distribution, we must get the evidence!
# 
# ### Evidence
# 
# The formula for computing the evidence is the following:
# 
# $$
# p(D) = \int\limits_{\theta}p(D|\theta)p(\theta)d\theta
# $$
# 
# Substituting the values and deriving the formula:
# 
# \begin{align}
# p(D) &= \int\limits_{0}^{1}p(D|\theta)p(\theta)d\theta\\
#      &= \int\limits_{0}^{1}(\theta)^h(1-\theta)^{N-h}\frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)}d\theta\\
#      &= \frac{1}{B(\alpha,\beta)}\int\limits_{0}^{1}(\theta)^{h+\alpha-1}(1-\theta)^{N-h+\beta-1}d\theta\\
#      &= \frac{1}{B(\alpha,\beta)}B(h+\alpha, N-h+\beta)\\
#      \therefore p(D) = \frac{B(h+\alpha, N-h+\beta)}{B(\alpha,\beta)}
# \end{align}
# 
# The last step follows from definition of [the Beta function](https://en.wikipedia.org/wiki/Beta_function).
# 
# ### Posterior
# 
# Now, we have all the required terms to compute the posterior $p(\theta|D)$.
# 
# \begin{align}
# p(\theta|D) &= \frac{p(D|\theta)}{p(D)}p(\theta)\\
# &= \theta^h(1-\theta)^{n-h} \cdot \frac{B(\alpha,\beta)}{B(h+\alpha, N-h+\beta)} \cdot \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)}\\
# &= \frac{\theta^{h+\alpha-1}(1-\theta)^{N-h+\beta-1}}{B(h+\alpha, N-h+\beta)}
# \\
# \therefore p(\theta|D) = Beta(h+\alpha, N-h+\beta)
# \end{align}
# 
# We have successfully derived the posterior and it follows a Beta distribution.
# 
# ## MAP is not the expected value of the posterior
# 
# From [Wikipedia](https://en.wikipedia.org/wiki/Beta_distribution), expected value of our posterior is:
# 
# $$
# \mathbb{E}_{\theta}(p(\theta|D)) = \frac{h+\alpha}{N + \alpha + \beta}
# $$
# 
# We derived the MAP as:
# 
# $$
# \theta_{MAP} = \frac{h+(\alpha-1)}{N+(\alpha-1)+(\beta-1)}
# $$
# 
# We can see that both values are clearly different. 
