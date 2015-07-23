# microscopes-lda

_A Python package for finding unobserved structure in unstructed data._

This package contains an implementation of the nonparametric (HDP) latent Dirichlet allocation (LDA) model described by Teh et al in [Hierarchal Dirichlet Processes](http://www.cs.berkeley.edu/~jordan/papers/hdp.pdf) (Journal of the American Statistical Association 101: pp. 1566–1581). Unlike the [original](https://www.cs.princeton.edu/~blei/papers/BleiNgJordan2003.pdf) LDA model, nonparametric LDA does not require the user to select a number of topics. Instead, the number of topics is inferred from the data using a hierarchal Dirichlet process prior.

The current kernel follows the sampling scheme described in Section 5.1 __Posterior sampling in the Chinese restaurant franchise__. In the future, we may support the other kernels described in Teh's paper.

Numerical computation is implemented in C++ for efficiency.

### Installation

OS X and Linux builds of `microscopes-lda` are released to [Anaconda.org](https://conda.anaconda.org). Installing them requires [Conda](https://store.continuum.io/cshop/anaconda/).  To install the current release version run:

```
$ conda install -c datamicroscopes -c distributions microscopes-lda
```