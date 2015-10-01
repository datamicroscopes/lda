# Change Log

## [Unreleased]
### Added

### Changed

### Fixed

### Removed

## [0.4.2]
### Added
- Add `active_dishes` method to state

### Fixed
- Bug where too many dishes are created when deserializing
- Bug fixed where dishes_ is constructed incorrectly when dish ids aren't contiguous

## [0.4.1]
### Added

### Changed

### Fixed
- Issue where deserialization sometimes failed due to deleted tables not being pruned from all vectors.

### Removed

## [0.3.1] - 2015-09-28
### Added
- Docstrings for most relevant/public methods
- Add serialization of state objects. They are also picklable.
- Make model_definition picklable.

### Changed
- Random number generator is no longer required for state object constructor

### Fixed
- Bug where perplexity calculation is assumed to be strictly monotone
- Bugs in explicit initialization. Now fully functional.
- Bug in sort order for term_relevance_by_topic

## [0.3.0] - 2015-09-16
### Added
- Add utility function for getting pyLDAvis data
- Added utility functions for translating data formats
- Wrote tests based on those in [ariddell's LDA](https://github.com/ariddell/lda/tree/57f721b05ffbdec5cb11c2533f72aa1f9e6ed12d/lda/tests)
- Added `term_relevance_by_topic` to get terms and relevance values as described by [Sievert and Shirley](http://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf)

### Changed
- New README
- Rename `word_distribution` method to `word_distribution_by_topic`
- Rename `document_distribution` method to `topic_distribution_by_document`.

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