# Change Log

## [Unreleased]
### Added
- Add utility function for getting pyLDAvis data

### Changed
- New README

### Fixed
- Multidish initialization actually works (from Python)
- Word distributions are normalized
- Word distributions map from word (hashable objects) to probabilities instead of index to probability.


## [0.2.2] - 2015-07-23
### Fixed
- Fixed bug where `initialize` incorrectly converted words (hashable objects) to integers (causing bad sampling issues).

## [0.2.1] - 2015-07-22
### Added
- Documents can now be any list-of-list of hashable objects
- Expose nwords method of state object to Python layer
- Can initialize state with more than one dish (topic)

### Removed
- Removed biology abstract test script and data

### Changed
- State object rolls up m_k instead of tracking m
- Reuse create_table and create_dish in state constructor
- Rename state.t_ji to state.table_doc_word

## [0.2.0] - 2015-07-20
### Added
- Initial (alpha) implementation HDP-LDA _Posterior sampling in the Chinese restaurant franchise_ (Section 5.1 in Teh, et al) based on [derivations](https://shuyo.wordpress.com/2012/08/15/hdp-lda-updates/) [done](https://github.com/shuyo/iir/blob/a6203a7523970a4807beba1ce3b9048a16013246/lda/hdplda2.py) by [Nakatani Shuyo](https://twitter.com/shuyo).

### Changed
- Currently uses vector of vector of integers to represent documents (instead of [variadic dataview](https://github.com/datamicroscopes/common/blob/master/include/microscopes/common/variadic/dataview.hpp)).
- Several changes to initialization API including hyperparameter setting.


## [0.1.0] - 2014-11012
### Added
- Initial attempt at HDP-LDA. Sampler does not work, however provides much of the code infrastructure for model.