---
title: "Stock Market Regime Detection and Forecasting using Hidden Markov Models and LSTM Networks"
subtitle: "Progress Submission: Literature Review and Exploratory Data Analysis"
author:
- "Ishraq Haque · Student Number 500947720"
- "MSc Data Science and Analytics, Toronto Metropolitan University"
- "Supervisor: Dr. Mohamed Wahab Mohamed Ismail"
date: "June 2026"
toc: true
toc-depth: 2
---

# Abstract

In this project I study whether identifying hidden market regimes can improve short-term financial forecasting. Financial markets move through different states over time, such as calm uptrends, turbulent selloffs, and range-bound periods, and these states are never labelled in the data. I treat them as latent states and estimate them with a Hidden Markov Model (HMM) applied to daily returns and volatility. I then feed the estimated regime into a sequence-learning forecaster to test whether that extra context improves the next-day prediction of returns and direction. I run the study on three markets that behave quite differently, the S&P 500 equity index, gold, and Bitcoin, so that my conclusions are not tied to a single asset class, and I use the VIX and U.S. Treasury yields as macro inputs. I frame the main question as a controlled experiment in which a regime-aware model and an otherwise identical regime-blind model differ only by the regime signal, and I plan to test the difference for statistical significance using a paired test across walk-forward folds. This submission reports two completed milestones. First, I review the literature on regime-switching models, deep-learning forecasting, and hybrid HMM-neural models, and I use that review to identify gaps that justify my design: limited testing across asset classes, a tendency to forecast price levels instead of returns, weak significance testing, and little comparison against modern attention-based models. Second, I carry out an exploratory analysis of more than three decades of daily data, which confirms the properties that motivate a regime-switching approach: heavy tails (excess kurtosis of 9.5, 9.0, and 12.4 for the S&P 500, gold, and Bitcoin), strong volatility clustering, and a clear rejection of normality for every asset. I close by describing my methodology and the planned experiments, and I provide the project repository.

**Keywords:** market regime detection; hidden Markov model; LSTM; financial time-series forecasting; volatility clustering; exploratory data analysis.

# 1. Introduction

Predicting how financial assets will move is one of the hardest problems in quantitative finance, because price series are noisy, non-stationary, and only weakly autocorrelated from one day to the next. One observation that comes up repeatedly in the literature is that markets do not behave the same way over time. The statistical properties of returns, including their mean, variance, and correlation structure, shift between distinct and persistent states that researchers call regimes [1]. A model fitted to the whole history averages over these regimes, and it can therefore do poorly at exactly the moments that matter most, such as the transition into a high-volatility crisis state.

This project looks at a direct way to deal with that problem. The idea is to estimate the current market regime explicitly and give it to a forecasting model as extra context. I treat regimes as latent (unobserved) states and recover them with a Hidden Markov Model, which is a probabilistic sequence model that is good at inferring hidden states from noisy observations [2]. I then pass the estimated regime to a Long Short-Term Memory (LSTM) network, and in an extension to a Temporal Fusion Transformer (TFT), to see whether the regime information produces a measurable improvement over an identical model that does not receive it.

My work is guided by four research questions:

1. **RQ1.** Can an HMM reliably identify distinct, interpretable market regimes from financial time series?
2. **RQ2.** Does adding regime information improve forecasting accuracy?
3. **RQ3.** How does my proposed hybrid model compare against standard baselines (ARIMA and a standalone LSTM)?
4. **RQ4.** Does a learned discrete regime representation capture market states better than a Gaussian HMM?

I treat this as a formal hypothesis test. The null hypothesis is that regime information does not improve performance, and the alternative is that it does. I plan to evaluate the hypothesis with a paired test (α = 0.05) on per-fold performance differences, so that any improvement I report rests on evidence rather than a single lucky run.

This document covers two completed parts of the project. Section 2 is my literature review and the critical analysis I use to derive the research gap and my design choices. Section 3 describes the dataset, its constraints, the preprocessing I applied, and my exploratory data analysis. Section 4 sets out the methodology and the planned experiments, with a diagram of the overall pipeline. Section 5 gives the repository link, and Section 6 lists the references.

# 2. Literature Review

My project draws on three research areas that grew up separately: regime-switching models in financial econometrics, deep learning for financial forecasting, and hybrid models that combine the two. A fourth and newer area, covering attention-based models and learned representations, motivates the extensions I plan. I review each area in turn and then give a critical analysis that sets out the gaps my project addresses.

## 2.1 Regime-switching models and latent market states

The formal study of regime changes in economic time series usually starts with Hamilton's Markov-switching model. He treated the parameters of an autoregression as the outcome of a discrete, unobserved Markov process, which gave a workable way to infer when an economy moves between states such as expansion and recession [3]. The model became a leading approach for dating business cycles and has been applied widely across macroeconomics and finance.

In finance, the Hidden Markov Model gives the same latent-state machinery in a form built for inference from noisy data. Rabiner's tutorial set out the standard algorithms, namely the forward-backward procedure, Viterbi decoding of the most likely state path, and Baum-Welch parameter estimation, and these remain the usual tools [2]. When an HMM is applied to returns and volatility, it recovers states that line up with recognisable conditions, such as low-volatility bull phases and high-volatility bear phases, and it produces both a decoded regime label and the posterior probability of each state at every point in time.

The reason for taking a regime-switching view is grounded in the empirical behaviour of returns. Cont's well-known survey of stylized facts shows that asset returns have heavy tails and positive excess kurtosis, almost no linear autocorrelation, and strong volatility clustering, where large moves tend to be followed by large moves and the autocorrelation of squared returns decays slowly [1]. A single fixed distribution cannot reproduce all of these features at once, but a mixture of regime-conditional distributions can, which gives a solid statistical reason to model returns as regime-dependent.

## 2.2 Deep learning for financial forecasting

The Long Short-Term Memory network of Hochreiter and Schmidhuber fixed the vanishing-gradient problem of earlier recurrent networks by using a gated memory cell, which let the model learn long-range dependencies in a sequence [4]. That ability matters for financial data, where useful structure can stretch across weeks or months.

The most influential large-scale study in finance is the one by Fischer and Krauss, who applied LSTM networks to next-day directional prediction of every S&P 500 constituent stock from 1992 to 2015 [5]. They found that LSTMs beat memory-free benchmarks such as random forests and standard feed-forward networks, reaching daily directional accuracy near 56% with positive risk-adjusted returns even after trading costs. Their result made the LSTM a strong and reproducible benchmark for daily forecasting, and it is the main reference point for the standalone-LSTM baseline I use.

## 2.3 Hybrid regime-aware deep learning

A growing set of studies combine regime detection with sequence learning, and this is the space my project sits in. Hu studied a combined HMM-LSTM approach on the S&P 500 index, using the HMM to describe market states and the LSTM to forecast, and reported that the combination improved forecasting [6]. Liu et al. analysed market trends with a similar HMM-LSTM pipeline, using the HMM to split the series into states that condition the later learning [7]. Li et al. proposed a modified LSTM (Mid-LSTM) aimed at anomaly-aware risk management, which shows how latent-state information can be folded in to improve robustness in turbulent periods [8]. The study closest to mine is by Jiang et al., who introduced a hidden-state-guided deep learning model (HMM-ALSTM) in which an HMM supplies state information that guides an attention-augmented LSTM, and reported gains on stock-movement forecasting over regime-blind versions [9]. Taken together, these papers give encouraging evidence that regime information can help, and they shape two of my own design choices: a sixty-day input window, which matches the convention in this literature, and feeding the regime in by concatenating the HMM posterior probabilities with the model's input features.

## 2.4 Attention-based architectures and learned representations

Two further developments motivate my planned extensions. The Transformer of Vaswani et al. replaced recurrence with self-attention, which lets a model relate distant positions in a sequence directly and exposes attention weights as a form of interpretability [10]. Building on that, Lim et al. proposed the Temporal Fusion Transformer (TFT), an attention-based model for multi-horizon forecasting that mixes recurrent local processing with interpretable self-attention, adds variable-selection gating, and separates static covariates, known-future inputs, and past-observed inputs [11]. The TFT comes with built-in variable-importance and temporal-attention outputs, which make it an appealing modern backbone for a study that cares about interpretability, and I use it as my second forecasting backbone.

On the representation side, van den Oord et al. introduced the Vector-Quantised Variational Autoencoder (VQ-VAE), which learns a discrete codebook of latent states through vector quantisation [12]. A VQ-VAE can therefore learn data-driven, non-Gaussian regime codes, unlike the parametric Gaussian states of a classical HMM. This motivates my fourth research question, which compares an HMM regime representation against a learned discrete one.

## 2.5 Critical analysis and research gap

The hybrid HMM-neural literature is encouraging, but reading it critically I see several recurring limitations, and my project is designed around them.

*Limited testing across asset classes.* Much of the hybrid work is shown on a single market, usually a major equity index such as the S&P 500 [6], [7]. Whether the benefit of regime information carries over to asset classes with very different dynamics, such as a low-volatility equity index, a safe-haven commodity, and a high-volatility cryptocurrency, is rarely tested. I test all three.

*Forecasting price levels instead of returns.* Several studies target price levels, where a naive predictor that just repeats yesterday's price can look very accurate because prices are strongly autocorrelated and non-stationary. That inflates the reported numbers and hides whether the model has any real skill. In line with the stylized-facts literature [1], I forecast next-day log-returns and direction, which are the right stationary targets for measuring forecasting ability.

*Weak significance testing.* Improvements credited to regime information are often reported as point differences in accuracy with no significance test, so it is not clear whether the gain is larger than ordinary run-to-run variation. I evaluate the regime contribution with a paired test across walk-forward folds and with multi-seed ensembles.

*Leakage risk in regime integration.* When regime states are estimated on the full sample and then used as model inputs, information from the future can leak into the training set. I fit the regime model on the training partition only and use causal, forward-filtered posteriors, so that every regime input respects the order of time.

*Few modern backbones and no learned regimes.* The hybrid literature is built almost entirely on recurrent backbones and Gaussian HMM states. It is largely unknown whether regime information still helps when the forecaster is a modern attention-based model [11], or whether a learned discrete regime [12] beats a Gaussian HMM. I address both by adding TFT-based models and a planned VQ-VAE regime detector.

Putting these points together, they justify my design: a controlled, leakage-safe, multi-asset comparison that isolates the value of regime information, tests it properly on stationary targets, and benchmarks it across both recurrent and attention-based backbones with classical and learned regime representations.

# 3. Data Description and Exploratory Data Analysis

## 3.1 Dataset summary

I use daily data from Financial Modeling Prep (FMP) under a paid subscription. I treat three assets as forecasting targets, chosen to cover different asset classes and risk levels: the **S&P 500** equity index, **gold** (spot, USD), and **Bitcoin** (USD). I use two further series as signals, meaning model inputs rather than forecasting targets, because they are well-known indicators of market stress: the **CBOE Volatility Index (VIX)** and **U.S. Treasury yields** at the 2-year and 10-year tenors, from which I derive the 10Y-2Y yield-curve slope. Table 1 summarises the coverage of each target series.

Table 1. Data coverage by target asset.

| Asset | First | Last | Observations |
|-------|-------|------|-------------:|
| S&P 500 | 1990-01-02 | 2026-06-03 | 9,172 |
| Gold | 1990-01-02 | 2026-06-04 | 9,304 |
| Bitcoin | 2014-01-01 | 2026-06-04 | 4,538 |

## 3.2 Constraints and preprocessing

A few data constraints shaped the preprocessing pipeline, which I built to be reproducible and leakage-safe.

- **Bitcoin start date.** Bitcoin's FMP history before 2014 is sparse and illiquid, with near-zero and unreliable prices. I therefore clip the series to start on 1 January 2014, so that I only analyse the period of genuine, liquid trading. I document this in the methodology.
- **Calendar alignment.** The target and signal series follow different trading calendars, since equities and Treasuries observe market holidays while Bitcoin trades every day. I align all series to a common index and forward-fill the macro signals onto the target calendar, so that a weekend cryptocurrency observation, for example, inherits the most recent known macro state. Forward-filling uses only past information, so it does not introduce look-ahead.
- **Return transformation.** I convert prices to daily log-returns. The forecasting target is the next-day log-return (a shift of one trading day), and direction is its sign, which keeps the target stationary and avoids the inflated accuracy that comes with predicting price levels.
- **Missing values and warm-up.** I drop rows that contain missing values from rolling-window warm-up periods, and I drop the final row of each series, which has no next-day target. This explains the gap between the raw observation counts in Table 1 and the smaller number of usable return observations in Table 2.
- **Volume excluded.** FMP volume coverage is inconsistent across the target series and missing for multi-year spans of the gold and Bitcoin histories. I exclude volume features by default so that I keep a uniform feature set and the largest usable sample.
- **Security.** I keep the FMP API key out of source control with an environment file, so the code can be released publicly in a safe way.

## 3.3 Descriptive statistics

Table 2 reports descriptive statistics for the daily log-returns of each target asset. These statistics put numbers on the stylized facts that motivate a regime-switching approach.

Table 2. Descriptive statistics of daily log-returns. Annualised volatility uses a 252-day convention; excess kurtosis is reported relative to the normal value of 0; *JB p* is the Jarque-Bera test p-value for normality.

| Asset | n | Mean | Std. | Ann. vol. | Skew | Excess kurt. | Min | Max | JB *p* |
|-------|----:|-----:|-----:|---------:|-----:|------------:|-----:|----:|-----:|
| S&P 500 | 8,451 | 0.0003 | 0.0113 | 0.180 | −0.17 | 9.49 | −0.100 | 0.110 | 0.000 |
| Gold | 8,623 | 0.0002 | 0.0104 | 0.165 | −0.45 | 8.97 | −0.121 | 0.088 | 0.000 |
| Bitcoin | 4,537 | 0.0010 | 0.0363 | 0.576 | −0.74 | 12.44 | −0.491 | 0.241 | 0.000 |

Three things stand out to me. First, the three assets sit in very different volatility ranges: annualised volatility rises from about 18% for the S&P 500 and 16% for gold to 58% for Bitcoin, which confirms the spread of risk levels I was aiming for. Second, all three series have large positive excess kurtosis (9.5, 9.0, and 12.4), which points to heavy tails far from a normal distribution, together with negative skewness that is strongest for Bitcoin. Third, the Jarque-Bera test rejects normality clearly for every asset (p ≈ 0).

## 3.4 Exploratory analysis and visualisation

In my exploratory analysis I look at the price histories, the return distributions, the dependence structure of volatility, and the macro context. Figures 1 to 9 give the visual evidence.

![Price history of the three target assets over the full sample. I plot each series on its own scale; the different trajectories of an equity index, a commodity, and a cryptocurrency are what motivate the cross-asset design.](../figures/fig01_price_history.png){width=95%}

![Empirical return distributions against a fitted normal density. The sharp peaks and heavy tails are visible for all three assets, which matches the large excess kurtosis in Table 2.](../figures/fig02_return_distributions.png){width=95%}

![Normal quantile-quantile plots of daily log-returns. The points pull away from the diagonal in both tails, which confirms non-normality and heavy tails.](../figures/fig03_qq_plots.png){width=95%}

![Absolute returns over time. The long bursts of large moves, separated by calm periods, are the visual signature of volatility clustering.](../figures/fig04_volatility_clustering.png){width=95%}

![Autocorrelation function of squared returns. The slow, persistent decay, which stays positive over many lags, is a numerical measure of volatility clustering. The lag-1 autocorrelation is 0.276 for the S&P 500, 0.113 for gold, and 0.142 for Bitcoin.](../figures/fig05_acf_squared_returns.png){width=85%}

![S&P 500 returns with the VIX overlaid. Spikes in the volatility index line up with the clustered high-volatility episodes, which links the macro stress signal to the target's conditional volatility.](../figures/fig06_vix_overlay.png){width=95%}

![The 10-year minus 2-year Treasury yield-curve slope. The curve was inverted (10Y below 2Y) on 11.6% of trading days since 1990, which is a recognised pre-recession signal and supports including the slope as a model input.](../figures/fig07_yield_curve_slope.png){width=95%}

![Rolling cross-asset return correlations. The correlations move over time rather than staying constant, which tells me that the right forecasting context itself shifts over time and supports a regime-aware design.](../figures/fig08_rolling_correlations.png){width=95%}

![Volatility terciles for the S&P 500. The index spends roughly a third of all days in the top volatility tercile, and these days bunch up inside the shaded crisis windows, which gives me a direct reason to use an explicit regime-switching model.](../figures/fig09_vol_regimes.png){width=95%}

## 3.5 Implications for modelling

My exploratory analysis leads to three conclusions that shape the modelling. First, the heavy tails and clear non-normality mean that a single Gaussian model is misspecified, whereas a mixture of regime-conditional distributions, which is what an HMM recovers, can fit the conditional return distribution better. Second, the strong and persistent volatility clustering tells me that latent, persistent states exist for a regime model to recover, which is a precondition for the HMM to be useful. Third, the time-varying cross-asset correlations suggest that the right forecasting context changes over time, and that is exactly the information a regime signal is meant to supply. Together these findings support my central hypothesis that regime context can improve multi-asset forecasting, and they justify the methodology I describe next.

# 4. Project Approach

## 4.1 Overview

I set the project up as a controlled experiment that isolates how much regime information contributes to short-horizon forecasting. The unit of comparison is a pair of models that are the same in every respect except that one receives the estimated market regime as an extra input and the other does not. Because the only difference between the two models is the regime signal, I can attribute any difference in forecasting performance to that signal. Figure 10 shows the full pipeline.

![Project methodology: the regime-aware forecasting pipeline, from raw data through preprocessing, feature engineering, regime detection, and forecasting to evaluation.](../figures/fig14_methodology.png){width=80%}

## 4.2 Feature engineering

From each target's price history I build fifteen leakage-safe features, where each feature on a given day uses only information available up to and including that day. They cover log-returns and short-horizon momentum, rolling volatility over several windows, distance-from-moving-average ratios, the Relative Strength Index, the Moving Average Convergence Divergence indicator, and the merged macro signals (VIX and the yield-curve slope, with their daily changes). I implement the technical indicators from first principles so that the feature set is fully auditable.

## 4.3 Regime detection

I infer market regimes with a Gaussian HMM fitted on standardised return and volatility features using the Baum-Welch algorithm [2]. I choose the number of states with the Bayesian Information Criterion, balanced against how interpretable the resulting regimes are. I fit the model on the training partition only and then apply it forward over the full series, which gives me, for each day, both a decoded regime and the causal, forward-filtered posterior probabilities of each state. I use the posterior probabilities, rather than the hard decoded label, as the model input, following the integration style of the earlier hybrid work [9]. As a planned extension that addresses RQ4, I will train a VQ-VAE [12] to learn a discrete regime representation from the same observations, which gives me a learned alternative to the Gaussian HMM that I can compare on equal terms.

## 4.4 Forecasting models

I compare three forecasting backbones under one shared evaluation protocol:

- **ARIMA.** A classical statistical baseline that gives the conventional reference point.
- **LSTM.** A recurrent network that learns from sixty-day input windows [4], in a standalone (regime-blind) form and in regime-aware forms.
- **TFT.** An attention-based model [11] that I use as a modern backbone and that provides built-in interpretability through variable-importance and temporal-attention outputs.

I evaluate each neural backbone with no regime input, with the HMM regime, and, as a planned extension, with the learned VQ-VAE regime. This gives a factorial comparison that separates the effect of the regime signal from the effect of the backbone.

## 4.5 Evaluation protocol

I designed the evaluation to be honest and leakage-safe. I split the data by time, training on the earlier period and testing on a later, unseen period, with no shuffling that could leak future information. I check robustness with expanding-window walk-forward folds, and I run the neural models under several random seeds and ensemble them to reduce run-to-run variance. I measure forecast quality with mean squared error, root mean squared error, and mean absolute error for the return forecast, and with directional accuracy and the F1 score for the up/down classification. I fit all scaling statistics on the training partition only, and I fit the regime model on the training data and apply it forward. I test the contribution of regime information for statistical significance with a paired t-test (α = 0.05) on per-fold differences between a regime-aware model and its regime-blind counterpart, so that any improvement I report reflects evidence rather than chance.

## 4.6 Justification of the approach

I ground each design choice in the literature from Section 2. The sixty-day input window and the concatenation of regime posteriors follow the conventions of the hybrid HMM-LSTM work [6], [7], [9]. My choice to forecast returns and direction instead of price levels follows directly from the stylized-facts literature [1], which shows that price-level prediction is confounded by non-stationarity. I use a Gaussian-mixture regime model because my exploratory analysis shows returns are heavy-tailed and non-normal, so a single Gaussian is misspecified. I add a modern attention backbone [10], [11] and a learned regime representation [12] to close the gap I identify in Section 2.5, which is that the hybrid literature has not shown whether regime information still helps beyond recurrent backbones and Gaussian states. Finally, the controlled, leakage-safe, significance-tested protocol responds to the methodological weaknesses I found in the earlier work.

## 4.7 Progress and next steps

The two milestones I report here, a literature review and a full exploratory data analysis with a reproducible data pipeline, are complete. My exploratory analysis confirms the empirical preconditions for a regime-switching approach. The next phases build the regime model and run the forecasting experiments I describe above, followed by the cross-asset evaluation and significance testing that will answer my four research questions.

# 5. GitHub Repository

I keep all code, configuration, processed-data definitions, and results for this project in a version-controlled repository:

**https://github.com/ishraqhaque/tmu-mrp-2026-ishraq-haque**

I organised the repository for reproducibility, with a configuration-driven pipeline, pinned dependencies, fixed random seeds, and the figures and findings I reproduce in this submission. I keep the API key needed for data collection out of source control.

# 6. References

[1] R. Cont, "Empirical properties of asset returns: stylized facts and statistical issues," *Quantitative Finance*, vol. 1, no. 2, pp. 223–236, 2001.

[2] L. R. Rabiner, "A tutorial on hidden Markov models and selected applications in speech recognition," *Proceedings of the IEEE*, vol. 77, no. 2, pp. 257–286, 1989.

[3] J. D. Hamilton, "A new approach to the economic analysis of nonstationary time series and the business cycle," *Econometrica*, vol. 57, no. 2, pp. 357–384, 1989.

[4] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.

[5] T. Fischer and C. Krauss, "Deep learning with long short-term memory networks for financial market predictions," *European Journal of Operational Research*, vol. 270, no. 2, pp. 654–669, 2018.

[6] D. Hu, "Forecast analysis of the stock market based on hidden Markov model and long short-term memory model, taking the S&P 500 index as an example," *Finance & Economics (Dean & Francis)*, 2024.

[7] M. Liu, J. Huo, Y. Wu, and J. Wu, "Stock market trend analysis using hidden Markov model and long short-term memory," *arXiv preprint* arXiv:2104.09700, 2021.

[8] X. Li, Y. Li, X.-Y. Liu, and C. D. Wang, "Risk management via anomaly circumvent: mid-LSTM," in *KDD Workshop on Anomaly Detection in Finance*, 2019.

[9] J. Jiang, L. Wu, H. Zhao, H. Zhu, and W. Zhang, "Forecasting movements of stock time series based on hidden state guided deep learning approach," *Information Processing & Management*, vol. 60, no. 3, art. 103328, 2023.

[10] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[11] B. Lim, S. Ö. Arık, N. Loeff, and T. Pfister, "Temporal fusion transformers for interpretable multi-horizon time series forecasting," *International Journal of Forecasting*, vol. 37, no. 4, pp. 1748–1764, 2021.

[12] A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural discrete representation learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.
