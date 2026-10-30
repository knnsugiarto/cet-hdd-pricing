# Defence sheet

Interview preparation for this project. Lines marked **[after the real run]** need the
numbers from `results/summary.md`; everything else holds whatever the results are.

## 60-second pitch

> I priced a hypothetical winter heating-degree-day option on the Met Office's Central England Temperature series, the kind of contract a gas utility uses to hedge cold winters. Temperature isn't traded, so there's no replication argument and no unique risk-neutral price. The price is the expected payoff under a model fitted to history, and the real question is whether the forecast distribution is calibrated. I compared three methods: burn analysis, a normal fit to seasonal totals, and a daily mean-reverting temperature model priced by Monte Carlo. I checked the Monte Carlo against an exact Gaussian closed form for the linear part of the index, season by season. I walked all three forward over 56 development winters, froze one variant per family by a rule written down in advance, and only then ran 20 holdout winters once. **[after the real run: one sentence on the headline, e.g. which method had the best CRPS and whether the difference was significant, and which pre-registered predictions held.]** The honest headline is about power: 20 winters can't detect a pricing bias smaller than most of the option's value, so I lead with distributional scores and report the detectable effect next to every pricing error.

## The ten questions you're most likely to get

**1. Why is there no risk-neutral price?**
Risk-neutral pricing comes from replication: you can hedge an equity option with the stock, so its price is pinned by no-arbitrage. You can't buy or store temperature, so no portfolio of traded assets replicates the payoff, the market is incomplete and many pricing measures are consistent with no arbitrage. Choosing one is equivalent to choosing a market price of weather risk, and with no traded CET contracts there's nothing to calibrate it to. So I price under the physical measure, which means assuming a zero market price of risk, and I show a loading $\alpha\,\mathrm{sd}$ as a sensitivity.

**2. What's the standard error of a mean payoff over N seasons, and what does it mean here?**
$\mathrm{sd}(\text{payoff})/\sqrt N$, if seasons are independent; I use Newey-West because a missed trend makes errors persist. For an at-the-money call on a normal index, the payoff's sd is about 1.46 times its price, so the SE of the mean pricing error is about $1.46/\sqrt N$ of the price: 33% at N = 20. A 5% test with 80% power needs a bias of about 2.8 SEs, roughly 90% of the price. A pricing-error test on 20 winters can only catch a model that is wildly wrong.

**3. Why CRPS rather than pricing error as the headline?**
CRPS scores the whole predictive distribution against the outcome and is proper: you can't improve your expected score by misreporting what you believe. It uses all of each outcome's information, not just whether it fell above a strike. And because every method faces the same winters, paired differences cancel the common noise, which gives far more power than comparing pricing errors one method at a time.

**4. Why should I believe this backtest?**
Five reasons, each checkable. (a) No look-ahead: the only function that sees temperatures gets them already cut at 31 October, and a test scrambles every later value and checks the forecasts don't change. (b) The selection rule was written down before any real-data run and is applied by code; the frozen variants are saved before the holdout is touched, and every holdout evaluation is logged with its commit and data hash. (c) All 24 variants are reported, not just the winners. (d) The predictions and their decision rules were committed in advance, and the verdicts are computed, not written. (e) The numerics are tested against closed forms, and a mutation check confirms the tests catch 14 specific bugs, including two look-ahead bugs. What it can't fix: the data are today's homogenised version, and I knew recent winters were mild. That's why there's also a forecast committed before its season started.

**5. Why does the Monte Carlo check use L instead of H?**
$H = \sum \max(18 - T_t, 0)$. If every day is below 18 °C, $H$ is a linear function of a Gaussian process and therefore Gaussian. But the fitted model puts a small probability on days above 18 °C in early November and late March, so $H = L + \sum \max(T_t - 18, 0) \ge L$ exactly. Comparing Monte Carlo on $H$ with a Bachelier price would eventually fail for a reason that isn't a bug. So I test the exact thing (the call on $L$, and $E[H]$ day by day) and report the gap between $H$ and $L$ separately. It's bigger at a 15.5 °C base, where warm days are more common.

**6. How do you get the variance of L in closed form?**
Write the AR(p) as a state vector $z_t = A z_{t-1} + b_t \varepsilon_t$ (companion form) and add the running sum $S_t = S_{t-1} + m_t X_t$, where $m_t$ is 1 on season days. The augmented system is still linear, so the mean and covariance propagate as $\mu_t = F_t \mu_{t-1}$ and $P_t = F_t P_{t-1} F_t^\top + g_t g_t^\top$, starting from the observed anomalies with zero covariance. For AR(1) this matches summing the explicit covariance matrix $\mathrm{Cov}(X_s, X_t) = \sigma^2 \phi^{|t-s|}(1-\phi^{2\min(s,t)})/(1-\phi^2)$, and a test checks that to machine precision.

**7. Isn't burn analysis just wrong when the climate is warming?**
Undetrended burn averages the last 30 winters, whose midpoint is 15 years back, so with warming it expects too many HDD. That's prediction P1. Detrending fixes the level but adds estimation noise from extrapolating a line. Which effect wins is an empirical question, and the development table answers it for this series. **[after the real run: say which won.]**

**8. Why did you pick those strikes?**
Every method must price the same contract, or the comparison means nothing. Strikes at the mean and ±1 sd of the previous 30 raw seasons are set the way a desk would quote around recent history, and they use only data before the pricing date.

**9. What if you had 100 times more data?**
With 2,000 holdout winters the power problem would disappear, and even small pricing biases would be detectable, though CET can't give that. The realistic routes to more power are monthly sub-indices (more, partly independent outcomes), many stations (more series, but correlated), or longer pre-1900 data (more history, but reconstructed and less reliable). Computation isn't the constraint: the whole study runs in about a minute.

**10. What would you do next?**
If P2 holds, replace the AR anomaly with a long-memory (ARFIMA) process or add a slow component; Caballero, Jewson and Brix showed this matters for weather indices. Then station data to measure basis risk, and meteorological forecasts for pricing close to and inside the season.

## Also likely

**How did you build it?** Answer truthfully. A version that is accurate: "I wrote the brief and made the design decisions, which are logged in `docs/decisions.md`. The code was written with an AI assistant. I ran every result on my machine, and I can walk you through any function." Only say the last part once it's true: work through the quizzes, and be able to derive the OU variance, the Bachelier formula and the Newey-West estimator unaided.

**Why AR(1) is an Ornstein–Uhlenbeck process.** $dX = -\kappa X\,dt + \sigma\,dW$ solves to $X_{t+\Delta} = e^{-\kappa\Delta}X_t + \text{Gaussian noise with variance } \sigma^2(1-e^{-2\kappa\Delta})/(2\kappa)$. Sampled daily, that is exactly an AR(1) with $\phi = e^{-\kappa}$.

## The weakest part, and how to talk about it

**Weakest part: statistical power.** Twenty winters cannot distinguish methods whose prices differ by less than most of an option's value, and one or two extreme winters dominate the holdout.

**How to say it:** "The honest result is about what 20 winters can and can't tell you. I calculated the detectable effect before running anything, led with the distributional scores, which have more power, and pre-registered what I expected so I couldn't reinterpret it afterwards. If I needed a sharper answer, I'd use monthly sub-indices or many stations, not more model complexity."

**Second weakest:** the short-memory daily model. If P2 holds, say so plainly: it's a known failure mode, the project measured it, and the fix is identified.
