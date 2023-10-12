# bw_temporalis Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### [0.9.2] - 2023-10-12

* Merge [#37](https://github.com/brightway-lca/bw_temporalis/pull/37) - Allow matrix edges without database `Exchange` objs

### [0.9.1] - 2023-10-12

* Fix multiplication of `FixedTD`

### [0.9] - 2023-10-11

* Allow division of absolute temporal distributions by numbers. Needed for unusual distributions in supply chain.

### [0.8.1] - 2023-10-10

* Add experimental node (instead of flow) timeline

### [0.8] - 2023-10-09

* Add `Timeline.add_metadata_to_dataframe` method
* Removed `utils.supplement_dataframe` function

### [0.7.2] - 2023-10-09

* Fix [#11](https://github.com/brightway-lca/bw_temporalis/issues/11) - Support uncertainty in graph traversal propagating to temporal distributions

### [0.7.1] - 2023-10-09

* Fix [#22](https://github.com/brightway-lca/bw_temporalis/issues/22) - Add a warning when the number of steps is much higher than the possible resolution in `easy` functions
* Fix [#10](https://github.com/brightway-lca/bw_temporalis/issues/10) - Support dynamic temporal distributions
* Fix [#7](https://github.com/brightway-lca/bw_temporalis/issues/7) - Add warnings to readme

## [0.7] - 2023-06-08

* Add `FixedTimeOfYearTD` and `FixedTD` distributions.
* Add structure for dynamic functions to override temporal distributions. Fixes [#10](https://github.com/brightway-lca/bw_temporalis/issues/10).
* Add ability to serialize and deserialize temporal distributions and dynamic distribution functions.

## [0.6] - 2023-05-08

* Fix [#15](https://github.com/brightway-lca/bw_temporalis/issues/15) - Switch `temporal_distribution` to have `amount` sum to one
* Fixes for [bw_graph_tools](https://github.com/brightway-lca/bw_graph_tools) version 0.2.2
* Lowered default cutoff values in graph traversal

## [0.5] - 2023-05-03

* New convolution implementation improves speed and removes dependency on `bw2speedups`
* Fix [#1](https://github.com/brightway-lca/bw_temporalis/issues/1) - Add utility function to check if temporal distributions sum to amount values
* Fix [#6](https://github.com/brightway-lca/bw_temporalis/issues/6) - Better error message for to_dataframe when timeline is empty
* Fix [#8](https://github.com/brightway-lca/bw_temporalis/issues/8) - Add clustering for to reduce complexity of large `TemporalDistributions` after convolution
* Fix [#9](https://github.com/brightway-lca/bw_temporalis/issues/9) - Add utility function to generate temporal distributions
* Fix [#12](https://github.com/brightway-lca/bw_temporalis/issues/12) - Add utility function to visualize temporal distributions

## [0.4] - 2023-04-29

* Can pass custom `graph_traversal` class to `TemporalisLCA`

## [0.3.1] - 2023-04-27

Bugfix release

## [0.3] - 2023-04-27

* Change exchange label from `temporal distribution` to `temporal_distribution`
* Add climate change characterization methods
* Add `supplement_dataframe` utility function

## [0.1.0] - 2023-04-26

Initial release

### Added

### Changed

### Removed
