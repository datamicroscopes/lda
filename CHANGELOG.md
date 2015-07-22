# Change Log

## [Unreleased]
### Added
- Documents can now be any list-of-list of hashable objects
- Expose nwords method of state object to Python layer

### Removed
- Removed biology abstract test script and data

### Changed
- State object rolls up m_k instead of tracking m

## [0.2.0] - 2015-07-20
### Added
- Initial (alpha) implementation HDP-LDA _Posterior sampling in the Chinese restaurant franchise_ (Section 5.1 in Teh, et al) based on [derivations](https://shuyo.wordpress.com/2012/08/15/hdp-lda-updates/) [done](https://github.com/shuyo/iir/blob/a6203a7523970a4807beba1ce3b9048a16013246/lda/hdplda2.py) by [Nakatani Shuyo](https://twitter.com/shuyo).

### Changed
- Currently uses vector of vector of integers to represent documents (instead of [variadic dataview](https://github.com/datamicroscopes/common/blob/master/include/microscopes/common/variadic/dataview.hpp)).
- Several changes to initialization API including hyperparameter setting.


## [0.1.0] - 2014-11012
### Added
- Initial attempt at HDP-LDA. Sampler does not work, however provides much of the code infrastructure for model.