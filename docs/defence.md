# Defence sheet

Interview preparation for this project. Numbers come from the first real-data run
(30 Oct 2026, commit `f1ad500`, `results/summary.md`); a re-run on a second machine reproduced them to one part in a billion.

## 60-second pitch

> I priced a hypothetical winter heating-degree-day option on the Met Office's Central England Temperature series, the kind of contract a gas utility uses to hedge cold winters. Temperature isn't traded, so there's no replication argument and no unique risk-neutral price. The price is the expected payoff under a model fitted to history, and the real question is whether the forecast distribution is calibrated. I compared three methods: burn analysis, a normal fit to seasonal totals, and a daily mean-reverting temperature model priced by Monte Carlo, checked against an exact closed form. I walked them forward over 56 development winters, froze one variant per family by a rule fixed in advance, then ran 20 holdout winters. All three pre-registered predictions held. Ignoring the warming trend overstated the index by about 65 heating degree days on the holdout, roughly half a standard deviation, with t = −2.3. The daily temperature model, the most sophisticated, was the worst calibrated: its 90% intervals covered only 77% of development winters, so it priced the at-the-money call at about half what the other methods did. And 20 winters couldn't separate the remaining methods, which I'd calculated before running anything.

## The ten questions you're most likely to get

**1. Why is there no risk-neutral price?**
Risk-neutral pricing comes from replication: you can hedge an equity option with the stock, so no-arbitrage pins its price. You can't buy or store temperature, so no portfolio of traded assets replicates the payoff. The market is incomplete, and many pricing measures are consistent with no arbitrage. Choosing one means choosing a market price of weather risk, and with no traded CET contracts there's nothing to calibrate it to. So I price under the physical measure, which means assuming a zero market price of risk, and I show a loading α·sd as a sensitivity. At α = 0.25 the loading adds about £290 to the burn method's at-the-money call on the holdout, about half its £607 average price.

**2. What's the standard error of a mean payoff over N seasons, and what does it mean here?**
sd(payoff)/√N if seasons are independent; I use Newey-West because a missed trend makes errors persist. For an at-the-money call on a normal index, the payoff's sd is about 1.46 times its price, so the SE of the mean pricing error is about 1.46/√N of the price: 33% at N = 20. A 5% test with 80% power needs a bias of about 2.8 SEs, roughly 90% of the price. The real holdout confirms it. The burn method's at-the-money call error is +£59 ± £356, with a detectable effect of about £1,000, against a price of about £600.

**3. Why CRPS rather than pricing error as the headline?**
CRPS scores the whole predictive distribution against the outcome, and it is proper: you can't improve your expected score by misreporting what you believe. It uses all of each outcome, not just which side of a strike it fell. And because every method faces the same winters, paired differences cancel the common noise. Even so, on the holdout the best method beats the baseline by 6.1 ± 8.9 HDD of CRPS: not significant.

**4. Why should I believe this backtest?**
Five reasons, each one checkable.
- **No look-ahead.** The only function that sees temperatures gets them cut at 31 October, and a test scrambles every later value and checks that no forecast changes.
- **The selection rule was fixed in advance and applied by code.** The frozen variants were saved before the holdout was touched, and the holdout log records one evaluation.
- **All 24 variants are reported,** not just the winners.
- **The predictions and their decision rules were written down before the run,** and the verdicts are computed, not written by hand.
- **The numerics are tested against closed forms,** a mutation check confirms the tests catch 14 specific bugs including two look-ahead bugs, and a re-run on a second machine matched every reported metric to one part in a billion.

What it can't fix: the data are today's homogenised version, I knew recent winters were mild, and the design commit was pushed after the run, so its timing rests on local timestamps. That is why there's also a forecast pushed before its season started.

**5. Your most sophisticated model did worst on calibration. Why?**
An AR(1) on daily anomalies has a persistence of about 0.81 per day, a half-life of about three days. Summed over 151 days, it predicts much less year-to-year variation than winters actually show: its forecast sd averaged about 106 HDD in development, against about 140 for the index model and a realised error RMS of about 137. Real temperatures have long memory, from slow drivers that a three-day memory can't represent. Caballero, Jewson and Brix showed in 2002, on three daily series including CET, that long-memory models price weather indices much more accurately than AR models; I pre-registered the consequence as P2. The fix is a long-memory (ARFIMA) or two-timescale anomaly model.

**6. AR(3) fixed the residual autocorrelation. Why didn't it help?**
The AR(1) residuals have lag-1 autocorrelation of about 0.13, and Ljung-Box rejects them outright. AR(3) whitens them (p ≈ 0.26). But AR(3) fits the next-day dynamics better by giving negative weight to the second lag, which lowers the long-run persistence that matters for a five-month sum. Its forecast sd fell to about 93 HDD and its 90% coverage to 70%. In-sample whiteness and out-of-sample calibration of a 151-day sum are different targets; I selected on the second.

**7. Isn't burn analysis just wrong when the climate is warming?**
Undetrended burn averages the last 30 winters, whose midpoint is 15 years back, so with warming it expects too many HDD. On the holdout it overstated the index by 64.7 ± 27.6 HDD, which overpriced calls and underpriced puts. All six of its option errors have the sign that bias predicts, and the puts' are significant: at the money, +£830 ± £266. These are six views of the same 20 outcomes, not six independent tests. Detrending removes the bias at the cost of extrapolating a line. In the development period the trend only paid off in the second half (1978–2005); in 1950–1977 no detrending scored better. That matches warming that accelerated after about 1980.

**8. How do you get the variance of L in closed form?**
Write the AR(p) in companion form, z_t = A z_{t−1} + b_t ε_t, and add the running sum S_t = S_{t−1} + m_t X_t, where m_t is 1 on season days. The augmented system is still linear, so the mean and covariance propagate as μ_t = F_t μ_{t−1} and P_t = F_t P_{t−1} F_tᵀ + g_t g_tᵀ, starting from the observed anomalies with zero covariance. For AR(1) this matches summing the explicit covariance matrix Cov(X_s, X_t) = σ² φ^|t−s| (1 − φ^(2 min(s,t))) / (1 − φ²), and a test checks that to machine precision. Over 76 seasons the Monte Carlo minus Bachelier z-scores had sd 0.97 and mean −0.20; five other seeds gave means between −0.19 and +0.12, so the −0.20 was chance.

**9. Why those strikes, and why is the baseline's at-the-money call so much dearer?**
Every method must price the same contract, so strikes are the mean and ±1 sd of the previous 30 raw seasons, set the way a desk quotes around recent history. That reference mean is exactly the baseline's own forecast mean, so for the baseline the call really is at the money, worth about 0.4 × sd × tick ≈ £1,100. The detrended methods put the expected index about 60 HDD below that strike, so the same call is out of the money for them and costs about half as much. That difference is the warming trend, priced.

**10. What would you do next, and what if you had 100 times more data?**
A long-memory anomaly model, because P2 is the clearest failure. Then station data to measure basis risk, and meteorological forecasts for pricing close to and inside the season. With 2,000 holdout winters the power problem would disappear. CET can't provide that, so the realistic routes are monthly sub-indices (more, partly independent outcomes) or many stations (more series, but correlated). Computation isn't the constraint: the full study runs in about 16 seconds on a laptop.

## Also likely

**How did you build it?** Answer truthfully. An accurate version: "I chose this project with help from an AI mentor and directed it. An AI assistant drafted the design and wrote the code, and every decision is logged with its alternatives. I ran the study on my machine, checked the results and can walk you through any function." Say the last part only once it's true: work through the quizzes, and be able to derive the OU variance, the Bachelier formula and the state-covariance recursion unaided.

**Why AR(1) is an Ornstein–Uhlenbeck process.** dX = −κX dt + σ dW solves to X_{t+Δ} = e^{−κΔ} X_t plus Gaussian noise with variance σ²(1 − e^{−2κΔ})/(2κ). Sampled daily, that is exactly an AR(1) with φ = e^{−κ}. Here φ ≈ 0.81, so κ ≈ 0.22 per day.

**Did the 15.5 °C check tell you anything?** No, and I say so in the decisions log. CET daily means essentially never reach 15.5 °C in winter, so the 15.5 °C index is the 18 °C index minus a constant.

## The weakest part, and how to talk about it

**Weakest part: statistical power.** Twenty winters can't distinguish methods whose prices differ by less than most of an option's value. Only the largest effects were detectable: the warming bias, which shows up in the baseline's put prices, and the daily model's narrow intervals.

**How to say it:** "The honest result is about what 20 winters can and can't tell you. I calculated the detectable effect before running anything, led with distributional scores because they have more power, and pre-registered what I expected so I couldn't reinterpret it afterwards. The warming bias and the daily model's under-dispersion were large enough to see; the differences between the good methods weren't. To sharpen that, I'd use monthly sub-indices or many stations, not a more complex model."

**Second weakest:** the pre-registration was committed locally before the run but pushed afterwards. Say so if asked, and point to the December forecast, which was pushed before its season started.
