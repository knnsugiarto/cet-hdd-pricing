# Decisions log

Each entry: what was chosen, the alternatives, and why. Entries D1 to D23 were written on
29 October 2026, **before any run on real data**. Anything changed after the first real
run gets a new, dated entry; nothing above is edited silently.

## D1 — Research question (29 Oct 2026)

- **Chose:** how well can a winter heating-degree-day (HDD) option on Central England Temperature be priced from temperature history alone? Compare burn analysis, index modelling and a daily stochastic model out of sample.
- **Alternatives:** a single "best" pricer; a trading backtest.
- **Why:** temperature is not traded, so the interesting question is forecasting the payoff distribution honestly, and the test is out-of-sample calibration. This is research: no result here is a return or a profit.

## D2 — Hypothetical contract (29 Oct 2026)

- **Chose:** daily HDD = max(18 °C − T, 0) on the CET daily mean; season 1 Nov to 31 Mar (151 or 152 days); index H = sum of daily HDD; tick £20 per HDD paid at season end; an HDD swap plus calls and puts at three strikes (D9). Every parameter lives in `configs/default.toml`.
- **Alternatives:** a 15.5 °C base (traditional in UK degree-day statistics); a station-based index.
- **Why:** 18 °C is the base the project brief specified; 15.5 °C is run as a robustness check, which also stresses the Gaussian approximation (D14). No exchange lists a CET contract, and CET is a regional composite rather than a station, so a real hedger would face basis risk. The README says so.

## D3 — What "price" means (29 Oct 2026)

- **Chose:** the expected payoff under the fitted physical model, which assumes a zero market price of weather risk. A risk loading E[payoff] + α·sd[payoff] is reported as a sensitivity (D11).
- **Alternatives:** calling it a risk-neutral or arbitrage-free price; calibrating a market price of risk.
- **Why:** there is no replicating portfolio for temperature, so the market is incomplete and there is no unique risk-neutral measure. There are no traded CET prices to calibrate a risk premium to.

## D4 — Data source and provenance (29 Oct 2026)

- **Chose:** Met Office Hadley Centre HadCET daily mean (`meantemp_daily_totals.txt`), downloaded by `scripts/download_data.py`, Open Government Licence v3. The run records the download time, the file's SHA-256 and a SHA-256 of the exact slice used (1 Nov 1900 to 31 Mar 2026).
- **Alternatives:** a hash of the whole file; the CEDA archive copy.
- **Why:** the live file gains a row every day, so a whole-file hash changes daily; the slice hash changes only if values actually used change. The CEDA archive needs an account and is not scriptable without one.

## D5 — Keep the latest winter in the holdout despite provisional values (29 Oct 2026)

- **Chose:** holdout = the 20 most recent complete winters, 2006/07 to 2025/26. January to March 2026 are provisional in the file ("v2.1.1.0 + provisional 2026 onwards"); the run lists every season that touches provisional values.
- **Alternative:** stop at 2024/25 so every holdout value is in a versioned release.
- **Why:** the brief asks for complete seasons, and 2025/26 is complete. How much provisional values are later revised is not known here; if a later version moves them, the slice hash changes and `results/holdout_runs.log` records the new hash.

## D6 — Pricing date (29 Oct 2026)

- **Chose:** each season is priced with data up to 31 October, the day before it starts. The daily model starts its simulation from the last observed anomalies.
- **Alternative:** price months ahead (as seasonal strips often trade).
- **Why:** it matches the brief's walk-forward rule. Starting from 31 October gives the daily model a small, legitimate information edge (a few days of persistence) that the other methods ignore; it is available information, so it is fair.

## D7 — Estimation start and the development/holdout split (29 Oct 2026)

- **Chose:** no data before 1 November 1900. Development seasons 1950/51 to 2005/06 (56). Holdout 2006/07 to 2025/26 (20).
- **Alternatives:** start in 1878 (when CET daily max/min begin); start in 1973 (the daily mean is anchored to Manley's monthly series before 1973).
- **Why:** 1900 leaves exactly 50 years before the first development season, so 50-year windows fit. Starting in 1878 adds older, more reconstructed data for little gain. Starting in 1973 would leave about three development seasons with 30-year windows. The 1973 construction change falls inside the estimation windows and is listed as a limitation.

## D8 — Leap days in burn and index methods (29 Oct 2026)

- **Chose:** put past seasons on the target season's calendar: drop 29 Feb when the target has none; when the target has one, add a proxy day for past non-leap seasons (the HDD of the mean of 28 Feb and 1 Mar).
- **Alternatives:** ignore the difference (one day is about 0.1 sd of H); rescale totals by season length.
- **Why:** cheap, exact for the calendar, and keeps a systematic one-day error out of the comparison. The realised index always uses the season's own calendar, as a contract would.

## D9 — Common strikes from recent history (29 Oct 2026)

- **Chose:** for season y, the strikes are m − s, m and m + s, where m and s are the mean and sd of the raw totals of the previous 30 seasons. The same strikes are used for every method.
- **Alternatives:** each method's own E[H] ± sd (the brief's wording); fixed strikes across all years.
- **Why:** with method-specific strikes the methods would price different contracts and could not be compared. Fixed strikes drift out of the money as winters warm. Quoting around recent history is how a desk would set strikes.

## D10 — Evaluate undiscounted prices (29 Oct 2026)

- **Chose:** compare realised payoffs with undiscounted expected payoffs (the forward value at the payment date). Discounting appears only in the forecast script's present value, at an assumed rate in the config.
- **Alternative:** discounted price against realised payoff.
- **Why:** a discounted price against an end-of-season payoff builds in a bias of (1 − DF) × price, about 1.5% over five months at 4%, unrelated to forecasting skill.

## D11 — Risk loading as a sensitivity (29 Oct 2026)

- **Chose:** price_α = E[payoff] + α·sd[payoff] for α ∈ {0, 0.1, 0.25}, reported for the frozen methods on the holdout and in the forecast.
- **Alternatives:** a utility-indifference price; a variance principle.
- **Why:** the standard-deviation principle is the common actuarial choice, needs no utility function, and shows directly how much of any pricing error a seller's margin would absorb.

## D12 — Index method's predictive sd includes estimation error (29 Oct 2026)

- **Chose:** sd = s·√(1 + 1/n) without trend, and s·√(1 + 1/n + (y₀ − ȳ)²/Sₓₓ) with a linear trend.
- **Alternative:** plug in the sample sd.
- **Why:** with 30 seasons the plug-in sd is too narrow, especially when the trend line is extrapolated a year beyond the data. Kept normal rather than Student-t for a closed-form price; the difference is small at n = 30.

## D13 — Daily model specification and grid (29 Oct 2026)

- **Chose:** T_t = s(t) + X_t with s(t) a linear trend plus 2 or 3 Fourier harmonics, fitted by least squares on the 30 or 50 years before the pricing date; X_t an AR(1) or AR(3) fitted by least squares without intercept; σ(t) constant or a 2-harmonic Fourier fit to squared AR residuals. 16 variants. 10,000 Monte Carlo paths per season and variant, with the same random draws for every variant in a season (common random numbers).
- **Alternatives:** AR(3) and seasonal volatility only if AR(1) diagnostics fail (the brief); a long-memory (ARFIMA) model.
- **Why:** with about 10,000 daily residuals a Ljung-Box test rejects almost any model, so diagnostics cannot be a clean gate. Running the small grid up front and counting it is more honest than a diagnostic-driven sequence. ARFIMA is the known fix for long memory (P2) but is out of scope for the time box; it is listed as the next step. Common random numbers make paired comparisons between daily variants less noisy.

## D14 — The Monte Carlo cross-check uses the linear index (29 Oct 2026)

- **Chose:** check the simulator against the exact Bachelier price of a call on L = Σ(base − T_t), which is Gaussian under the model, and the simulated mean of H against the exact E[H] (a sum of per-day Gaussian expectations). Report the truncation gap (call on H minus call on L, same paths) separately.
- **Alternative:** compare Monte Carlo on H directly with a Bachelier price (the brief's original wording).
- **Why:** H = L + Σ max(T − base, 0) ≥ L. The model puts a small probability on days above the base in early November and late March, so H is not exactly Gaussian, and with enough paths a direct comparison would fail for a reason that is not a bug. The linear-index check is exact; the gap measures the approximation. It is larger at 15.5 °C.

## D15 — Frozen selection rule (29 Oct 2026)

- **Chose:** in each family, the variant with the lowest mean development CRPS at 18 °C is frozen. The holdout evaluates only the frozen variants and the baseline (naive burn: no trend, 30 seasons). `results/frozen.json` is written before any holdout season is forecast, and every holdout evaluation is appended to `results/holdout_runs.log` with the commit and data hash.
- **Alternatives:** choose by hand after looking at development plots; evaluate every variant on the holdout.
- **Why:** an automatic, pre-declared rule cannot be bent after seeing results. Evaluating all 24 variants on the holdout would invite picking the best one there.

## D16 — Pre-registered predictions and their decision rules (29 Oct 2026)

Written before any real-data run. The verdicts in the README are computed by `hddlab.evaluate.verdicts` with exactly these rules.

- **P1.** Undetrended burn overestimates E[H] in recent decades because winters have warmed. *Rule:* on the holdout, the baseline's mean of (realised H − forecast mean) has a Newey-West t below −1.645. Development seasons from 1980 are shown as supporting evidence only.
- **P2.** The daily AR model's predictive sd of H is too small, so its 90% intervals under-cover. Reason: short-memory AR models forget within days and understate year-to-year variability; Caballero, Jewson and Brix (2002) showed long-memory models price weather indices much more accurately than AR models on daily series including CET. *Rule:* for the frozen daily variant on development seasons, 90% coverage is below 0.9 by more than 1.645 binomial standard errors.
- **P3.** On 20 holdout seasons, no pair of frozen methods differs significantly in how well it prices the at-the-money call. *Rule:* for every pair, the paired Newey-West t-statistic of the difference in squared pricing errors has |t| < 1.96.

*Refinement of P3 relative to the scope message (still before any data):* the scope said "no pairwise pricing difference is significant". Differences of pricing errors between two methods are just differences of their prices, which contain no outcome noise, so that test would be meaningless. Squared error is a consistent score for an expected value, so the test compares squared pricing errors.

## D17 — Headline metrics and statistical power (29 Oct 2026)

- **Chose:** headline = CRPS of H (lower is better) as paired differences against the baseline; plus bias and RMSE of E[H], interval coverage, PIT histograms and the RMS of standardised errors. The option pricing error (payoff − price) is reported with its standard error and minimum detectable effect (MDE: 5% two-sided test, 80% power, 2.80 standard errors).
- **Alternative:** lead with the mean pricing error.
- **Why:** for an at-the-money call on a roughly Gaussian H, sd(payoff) ≈ 1.46 × price regardless of sd(H). With N seasons the MDE is about 2.8 × 1.46/√N of the price: roughly 90% of the price on 20 holdout seasons and 55% on 56 development seasons. The pricing metric cannot carry the headline; the distributional metrics use the whole outcome, and paired differences cancel the common outcome noise.

## D18 — Newey-West standard errors with 3 lags (29 Oct 2026)

- **Chose:** every standard error of a mean across seasons uses Newey-West with Bartlett weights and 3 lags.
- **Alternatives:** naive SE; block bootstrap.
- **Why:** a trend that a method misses produces runs of same-signed errors, so naive SEs would be too small. Three lags follows the usual rule of thumb (about 4(N/100)^(2/9)) for 20 to 60 observations.

## D19 — Pre-registered forecast of the December 2026 to March 2027 index (29 Oct 2026)

- **Chose:** `scripts/forecast.py make --season-start 2026-12-01 --season-end 2027-03-31`, run with the frozen variants and committed to git before 1 December 2026. `scripts/forecast.py score` scores it once March 2027 data are published.
- **Alternatives:** a full-season forecast for 2026/27 (it starts 1 November, two days after this entry, before the models could be frozen); skip pre-registration.
- **Why:** a git timestamp before the season proves the forecast was not fitted to the outcome. One season is a single draw: it demonstrates discipline, not accuracy.

## D20 — Daily diagnostics are reported, not used for selection (29 Oct 2026)

- **Chose:** residual autocorrelations at lags 1 to 5 (mean over development seasons) and a Ljung-Box test on the last development fit are reported; selection uses only out-of-sample CRPS.
- **Why:** see D13. In-sample whiteness and out-of-sample skill can disagree; the question is about the latter.

## D21 — Mutation check (29 Oct 2026)

- **Chose:** `scripts/mutation_check.py` injects 14 specific bugs (HDD sign, leap year, look-ahead in the burn window and the cutoff, the variance recursion, the AR lag alignment, and others) into a temporary copy and confirms that the test suite fails for each.
- **Why:** it turns "the tests would catch this" into a reproducible result.

## D22 — Configuration in TOML; no demo app (29 Oct 2026)

- **Chose:** `configs/default.toml` read with the standard library's `tomllib`; no Streamlit demo.
- **Alternatives:** YAML (one more dependency); a Streamlit app.
- **Why:** fewer dependencies. A quant research reviewer reads the results and the code, not a demo.

## D23 — Variants counted (29 Oct 2026)

- **Chose:** 24 variants (8 burn/index, 16 daily), each also evaluated at 15.5 °C for robustness only. Every variant appears in `results/summary.md`; none is hidden.
- **Why:** with 24 variants, the best development score is optimistic by construction. That is why the holdout uses only frozen variants.

---

Entries below were written **after** the first real-data run (30 Oct 2026, 00:56 UTC, commit `f1ad500`). None changes a modelling choice or the frozen variants.

## D24 — Recording the code version correctly; when the design was published (30 Oct 2026)

- **What happened:** the first run logged its commit as `f1ad500+dirty`. The "dirty" flag came from the run's own output: `results/frozen.json` was written, untracked, before the code checked `git status`. The results commit `55b62a7` adds only files under `results/` and the generated README block on top of `f1ad500`, so the code that ran is exactly `f1ad500`. Re-running the study from the downloaded file on a second machine (Linux) matched every number in `results/metrics.json` to one part in a billion.
- **Chose:** record the git state before the run writes anything, and ignore `results/` when deciding "dirty". A test covers both cases.
- **Also recorded:** the design commit (`f1ad500`, 00:55:40 UTC) precedes the data download (00:55:46) and the run (00:56), but it was pushed to GitHub only after the run. The order rests on local git timestamps, which can be forged, so it is weaker evidence than a public push. The December 2026 to March 2027 forecast (D19) must be pushed before 1 December to give a public timestamp.

## D25 — The 15.5 °C robustness check is uninformative on CET (30 Oct 2026)

- **Finding:** no CET daily mean reached 18 °C in any November to March from 1900/01 to 2025/26, and only one day reached 15.5 °C. When T stays below the base, max(b − T, 0) = b − T, so the 15.5 °C index is the 18 °C index minus 2.5 °C × season length, a constant (apart from that one November day). The scores and pricing errors are essentially unchanged, and the truncation gap of D14 is negligible at either base.
- **Chose:** keep the check (it was pre-declared) and say plainly that it tests nothing here. D2's claim that the lower base "stresses the Gaussian approximation" holds in general but not for this series.
- **Alternative not taken:** add a new robustness check after seeing the results. That would be a post-hoc choice; if one is added later, it gets its own dated entry and is labelled exploratory.

## D26 — Monte Carlo z-scores averaged −0.20 (30 Oct 2026)

- **Finding:** over 76 seasons, the frozen daily variant's Monte Carlo minus Bachelier z-scores had mean −0.20 (standard error about 0.11). Re-running with five other seeds gave means of +0.10, +0.05, +0.12, −0.19 and +0.03.
- **Conclusion:** chance, not bias. Every run before this check used the same seed, so the same draws produced the same −0.20 each time. No change.
