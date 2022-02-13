#!/usr/bin/env python
# coding: utf-8

# # Bayesian Framework

# <!-- <textarea> -->
# The entire Bayesian modeling is based on the Bayes rule stated as:
# 
# \begin{equation}
# p(B|A) = \frac{p(A|B)}{p(A)}p(B)
# \end{equation}
# 
# But, what is A and B? well, it depends on our need. As long as it follows the principles of math, we can model anything using the Bayes rule. 
# 
# In classification and regression problems, our focus is to model the outputs in form of probability distributions. We will mainly look into two kinds of modeling.
# 
# (parameters-framework)=
# ## 1. Modeling the parameters (e.g. coin toss, linear regression)
# 
# Here, we will assume a prior distribution (mostly continuous) over the model parameters $\theta$ and derive a posterior distribution of parameters conditioned on the observed data.
# 
# \begin{equation}
# \underbrace{p(\theta|D)}_{\text{Posterior}} = \frac{\overbrace{p(D|\theta)}^{\text{Likelihood}}}{\underbrace{p(D)}_{\text{Evidence}}}\underbrace{p(\theta)}_{\text{Prior}}
# \end{equation}
# \begin{equation}
# p(D) = \int_{\theta}p(D|\theta)p(\theta)d\theta
# \end{equation}
# 
# In this type of modeling, distribution over the outputs may be derived from the predictive distribution of parameters.
# 
# (outputs-framework)=
# ## 2. Modeling the outputs (e.g. Gaussian processes)
# 
# Here, we will assume a prior distribution over the outputs (functions) $\mathbf{f}$. Posterior distribution is the distribution over the outputs conditioned on observed data.
# 
# \begin{equation}
# \underbrace{p(\mathbf{f}|D)}_{\text{Posterior}} = \frac{\overbrace{p(D|\mathbf{f})}^{\text{Likelihood}}}{\underbrace{p(D)}_{\text{Evidence}}}\underbrace{p(\mathbf{f})}_{\text{Prior}}
# \end{equation}
# \begin{equation}
# p(D) = \int_{\mathbf{f}}p(D|\mathbf{f})p(\mathbf{f})d\mathbf{f}
# \end{equation}
# 
# Don't worry about the details if you got the bigger picture for now.
# 
# Let's move to the simplest example in probability, a coin toss experiment!
# <!-- </textarea> -->
