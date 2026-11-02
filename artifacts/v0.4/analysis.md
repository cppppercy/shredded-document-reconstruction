# V0.4 independent ablation and confirmation

## Protocol

- Training families: 24 new V0.4 families (seeds 9100+); validation families: 12 (10100+); confirmation families: 24 (11100+).
- Every family is evaluated under five fixed observations (`clean`, `no_assists`, `scan`, `edge_loss`, `vertical_shift`), yielding 120 confirmation cases.
- The five configurations were fixed before confirmation. The validation table is retained as a descriptive check; no parameter was selected from confirmation results.
- `traditional_no_endpoint` removes only the traditional endpoint auxiliary cost. `learned_v03_no_augmentation` sees clean observations only. `learned_v03_without_alignment` retains nuisance augmentation but uses V0.1 features and zero vertical search. `learned_v03_full` uses the V0.3 feature/alignment/augmentation configuration with endpoint multiplier fixed at 0.0.

## Confirmation results

| Method | Document-mean adjacency | Exact recovery |
| --- | ---: | ---: |
| Traditional V0.2 | 78.54% | 47.50% |
| Traditional without endpoint cost | 72.96% | 1.67% |
| V0.3 learned, no nuisance augmentation | 77.39% | 48.33% |
| V0.3 learned, no alignment feature/search | 76.28% | 40.00% |
| V0.3 learned, full | 77.78% | 45.00% |

The full-model differences relative to no-augmentation and no-alignment variants are respectively +0.39% and +1.50%. These are confirmation-set observations, not tuning decisions.

## External pages

New sources are full public pages followed by simulated vertical cuts, not physical-shred scans. Six pages from three new sources were held outside every model split. The complete metrics and provenance are in `external_evaluation.csv` and `external/external_sources.json`.

## Saved reconstruction cases

- `success-v04-report-8-clean-11100.png`
- `failure-v04-table-10-vertical_shift-11211.png`
- `success-v04-usgs-landstat-economic-valuation-10-external_page_simulated_cut-usgs-landstat-economic-valuation-9.png`
- `failure-v04-gpo-annual-report-2023-10-external_page_simulated_cut-gpo-annual-report-2023-4.png`
