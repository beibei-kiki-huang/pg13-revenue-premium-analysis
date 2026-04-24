# The $159M Question: PG-13 vs R
### Movie Rating Revenue Analysis | USC Marshall DSO 510

![Python](https://img.shields.io/badge/Python-Regression_Analysis-blue)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## Research Question

For high-budget blockbusters, does a family-friendly rating (PG-13) generate a
statistically significant revenue premium compared to an R-rating?

**H₀:** PG-13 movies do NOT generate higher box-office revenue than R-rated movies.  
**H₁:** PG-13 movies DO generate higher box-office revenue than R-rated movies.

---

## Key Finding

> **A PG-13 rating independently adds $159 million to a film's global box office
> compared to an R-rating** — after controlling for budget, genre, runtime,
> year, and seasonality.

---

## Regression Model

**Dependent Variable:** Worldwide Gross Revenue  
**Main Independent Variable:** MPAA Rating (baseline = R)  
**Controls:** Production Cost, Genre, Runtime, Year, Seasonality

| Variable | Coefficient | p-value |
|---|---|---|
| PG-13 Rating | +$159.1M | 0.001 |
| Production Cost | 3.55x | 0.000 |
| Runtime | +$5.23M/min | 0.000 |
| Seasonality | — | > 0.05 |

**R-squared:** 0.371

---

## Prediction Case Study — Skyfall (2012)

| | Value |
|---|---|
| Inputs | Budget $200M, Action, PG-13, 2012 |
| Predicted Revenue | $0.79B |
| Actual Revenue | $1.11B |
| 95% Prediction Interval | [$0.16B — $1.42B] |

Model was conservative (underestimating Bond franchise power), but actual
revenue falls within the 95% prediction interval — validating model reliability.

---

## Key Insights

- PG-13 films scale better per dollar of budget (steeper trend line vs R)
- Seasonality becomes insignificant after controlling for budget — summer
  blockbusters earn more because they spend more, not just because of timing
- Group entertainers contribute 88% of revenue (entertainment agency parallel)
- Supply gap identified: Modern Rock, 80's Music — demand exists, no supply

---

## Files

| File | Description |
|---|---|
| `analysis.py` | Full Python regression + visualization code |
| `top-500-movies.csv` | Dataset (Kaggle: Top 500 Movies by Production Budget) |
| `BA Final Presentation.pdf` | Executive presentation deck |

---

## Team — USC Marshall DSO 510 (Fall 2025)
Beibei (Kiki) Huang · Yuanyan (Leonor) Jiang · Molly Ko · Zheyu (Alan) Deng ·
Apoorva Kamath · Jace Kang · Wei Huang · Connor Buck
