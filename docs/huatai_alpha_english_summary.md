# Research Summary

**Revisiting Price–Volume Alpha Factors in China's A-Share Market**

This project reproduces Alpha3, Alpha13 and Alpha44 from Huatai Securities' 2019 research and examines their predictive properties in recent Chinese equities. The original formulas, signs and lookback windows are preserved. The study emphasizes empirical replication, distributional diagnosis and reproducibility rather than maximizing historical returns.

Approximately 2.8 million daily records cover 1,263 historical CSI 500 securities. Dynamic membership reconstructed from monthly snapshots is explicitly treated as approximate point-in-time information. Signals span 2023–2025; the principal five- and twenty-market-day comparisons use 707 common dates through December 3, 2025. Prices before index entry and after exit remain available, and market-calendar endpoints determine future returns. Missing quotes are not converted into ordinary trading observations or zero-return labels.

Alpha13's five-day mean RankIC is approximately 0.0199, with a nonannualized ICIR of 0.258. Alpha44's corresponding values are approximately 0.0214 and 0.248. Both five-day mean-IC tests survive the prespecified eight-test Holm correction, whereas twenty-day evidence is weaker. However, positive rank correlation does not guarantee increasing arithmetic portfolio returns.

Independent checks find no reversed sign, incorrect quantile order or mismatch between the observable IC sample and the union of quantile samples. Alpha13's five-day Q5-minus-Q1 arithmetic return spread is −1.64 basis points, compared with a positive daily-median spread of 25.14 basis points and a 1st/99th-percentile trimmed spread of 3.15 basis points. Strong positive-return tails in Q1 raise its arithmetic mean. Individual stock contributions, annual variation, industry exposures and index-reconstitution windows also matter. These are diagnostic decompositions, not retrospective stock exclusions. Alpha44's positive 5.28-basis-point spread is less adversely affected by tails, but its spread is not statistically robust and is not positive in every year.

Residualizing Alpha ranks against contemporaneous MOM20 ranks leaves positive five-day conditional ranking evidence. This does not establish causality, nonlinear independence or reliably tradable incremental profits. All negative and nonsignificant findings are retained.

The prespecified Alpha13 twenty-day long-only illustration returns approximately 7.96% annualized under a 15-basis-point fee per side, with −0.49% cumulative relative performance against the CSI 500 price index. Complete five-day and twenty-day diagnostic comparisons are reported without selecting an optimal strategy. The simplified next-open ledger blocks missing quotes and opening price limits, charges actual modeled trades and reconciles cash and holdings. Auction liquidity, exact corporate-action cash flows and delisting settlements remain unresolved.

Tests, independent scalar calculations, preserved input hashes and numerical comparisons before and after refactoring support reproducibility. Approximate information timing, previously explored 2024 data, missing outcomes and the short sample remain important limitations. The project illustrates why formula correctness, statistical ranking ability and economic investment value require separate verification.
