# P-Pinpoint

> **Team Repo:** [github.com/KazumaOkada36/P-Pinpoint](https://github.com/KazumaOkada36/P-Pinpoint)
> **Duration:** 10 Weeks
> **Goal:** Build a web application that recommends optimal business locations based on company size, stage (startup, pharma, enterprise, etc.), and relevant geographic/economic features — powered by a neural network backend and an interactive frontend.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Frontend (React)                  │
│  Input Form → Interactive Map → Results Dashboard   │
└──────────────────────┬──────────────────────────────┘
                       │  REST API (Flask / FastAPI)
┌──────────────────────▼──────────────────────────────┐
│                    Backend (Python)                  │
│  Data Pipeline → Feature Engineering → ML Model     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Database / Data Store                   │
│  Location features, company profiles, model weights │
└─────────────────────────────────────────────────────┘
```

---

## Week-by-Week Timeline

### Phase 1 — Foundation (Weeks 1–3)

#### Week 1: Project Scoping & Data Strategy
| Task | Owner | Deliverable |
|---|---|---|
| Define target company categories (startup, pharma, tech, manufacturing, etc.) | All | Spec document |
| Identify key location features (cost of living, talent pool density, tax incentives, proximity to universities/labs, infrastructure, etc.) | All | Feature list |
| Survey & collect datasets (Census Bureau, BLS, Zillow, university locations, NIH/biotech hubs, etc.) | Data team | Raw data in `Data_Set/` |
| Set up repo structure: `/frontend`, `/backend`, `/data`, `/models`, `/docs` | All | Clean repo |
| Set up dev environment & CI (linting, basic test scaffold) | All | Working local setup |

#### Week 2: Data Collection & Cleaning
| Task | Owner | Deliverable |
|---|---|---|
| Scrape / download location datasets (city-level or zip-code-level) | Data team | CSVs in `Data_Set/` |
| Clean and normalize data (handle missing values, standardize units) | Data team | `clean_data.py` pipeline |
| Exploratory data analysis — distributions, correlations, outliers | Data team | EDA notebook (`notebooks/eda.ipynb`) |
| Define the **label/target**: e.g., composite "suitability score" per location-company-type pair, or cluster-based ranking | All | Documented labeling strategy |
| Begin frontend wireframes (Figma or sketches) | Frontend | Wireframe images / Figma link |

#### Week 3: Feature Engineering & Frontend Scaffold
| Task | Owner | Deliverable |
|---|---|---|
| Engineer features: normalize, encode categoricals, create composite indices (e.g., talent index, cost index, infrastructure index) | Data team | `feature_engineering.py` |
| Build train/validation/test splits | Data team | Split datasets |
| Scaffold React app: routing, layout, placeholder pages (Home, Input Form, Results) | Frontend | Running React app |
| Set up backend server (Flask or FastAPI) with a health-check endpoint | Backend | `app.py` running on localhost |

---

### Phase 2 — Model Development & Core UI (Weeks 4–6)

#### Week 4: Baseline Model
| Task | Owner | Deliverable |
|---|---|---|
| Implement a **baseline model** (e.g., weighted scoring / logistic regression / random forest) to establish benchmark performance | ML team | `models/baseline.py`, logged metrics |
| Design neural network architecture (MLP or embedding-based) that takes company features + location features → suitability score | ML team | Architecture diagram |
| Build the input form on frontend: company name, size (slider/dropdown), stage (startup/pharma/enterprise), industry | Frontend | Working form component |
| Define REST API contract: `POST /predict` with request/response schema | Backend + Frontend | API spec (OpenAPI or doc) |

#### Week 5: Neural Network Training
| Task | Owner | Deliverable |
|---|---|---|
| Implement neural network in PyTorch or TensorFlow | ML team | `models/neural_net.py` |
| Train on prepared dataset; tune hyperparameters (learning rate, layers, dropout) | ML team | Training logs, loss curves |
| Compare NN vs. baseline — decide whether NN adds value or pivot to ensemble/gradient boosting | ML team | Comparison report |
| Build interactive map component (Leaflet.js or Mapbox) on frontend | Frontend | Map rendering with dummy pins |
| Implement `/predict` endpoint that loads model and returns top-N locations | Backend | Working endpoint |

#### Week 6: Integration V1
| Task | Owner | Deliverable |
|---|---|---|
| Connect frontend form → backend API → model inference → map display | All | End-to-end demo working locally |
| Display results: ranked list of cities/regions with scores + map pins | Frontend | Results dashboard |
| Add loading states, basic error handling | Frontend | Polished UX |
| Write model evaluation script (accuracy, ranking metrics like NDCG or MAP) | ML team | `evaluate.py` |
| **Milestone: Internal demo / midpoint review** | All | Working prototype |

---

### Phase 3 — Refinement & Polish (Weeks 7–9)

#### Week 7: Model Improvement & Advanced Features
| Task | Owner | Deliverable |
|---|---|---|
| Incorporate additional data sources or features based on midpoint feedback | Data team | Updated dataset |
| Experiment with model improvements: feature importance analysis (SHAP), architecture tweaks, or alternative algorithms (XGBoost, ensemble) | ML team | Improved model metrics |
| Add explainability to results: "Why this location?" — show top contributing factors per recommendation | Backend + Frontend | Explanation cards in UI |
| Add filters/preferences: budget range, preferred region, proximity to airports/universities | Frontend | Advanced input options |

#### Week 8: UI Polish & Edge Cases
| Task | Owner | Deliverable |
|---|---|---|
| Responsive design — ensure works on mobile and tablet | Frontend | Mobile-friendly UI |
| Side-by-side location comparison feature | Frontend | Comparison view |
| Handle edge cases: unusual inputs, no results, model uncertainty | Backend | Graceful error responses |
| Add data visualizations: bar charts for factor breakdown, radar/spider charts for location profiles | Frontend | Chart components (Recharts / D3) |
| Write unit & integration tests for API and model inference | All | Test suite (pytest + Jest) |

#### Week 9: Deployment & Performance
| Task | Owner | Deliverable |
|---|---|---|
| Containerize app (Docker) | Backend | `Dockerfile`, `docker-compose.yml` |
| Deploy backend (Render, Railway, AWS, or GCP) | Backend | Live API endpoint |
| Deploy frontend (Vercel, Netlify, or GitHub Pages) | Frontend | Live website URL |
| Performance optimization: model inference caching, lazy-load map tiles | All | <2s response time target |
| Load testing & stress testing | Backend | Performance report |

---

### Phase 4 — Launch (Week 10)

#### Week 10: Final Testing, Documentation & Presentation
| Task | Owner | Deliverable |
|---|---|---|
| End-to-end QA: test all user flows with real inputs | All | Bug-fix log |
| Write `README.md` with setup instructions, architecture overview, screenshots | All | Updated repo README |
| Create project documentation: data sources, model methodology, API docs | All | `/docs` folder |
| Prepare demo presentation / video walkthrough | All | Slide deck + demo |
| **Final launch / submission** | All | 🚀 Live product |

---

## Key Milestones

| Week | Milestone |
|---|---|
| 3 | Data pipeline complete, frontend scaffold running, backend server live |
| 6 | **Working prototype** — end-to-end form → model → map results |
| 9 | Deployed to production, polished UI, tested |
| 10 | **Final launch** — documented, presented, live |

---

## Suggested Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + TypeScript, Leaflet.js or Mapbox GL, Recharts/D3 |
| Backend | FastAPI (Python), Pydantic for validation |
| ML/Model | PyTorch (or scikit-learn/XGBoost as fallback) |
| Data | Pandas, NumPy, SHAP for explainability |
| Database | SQLite (dev) → PostgreSQL (prod), or simply CSV/Parquet files |
| Deployment | Docker, Vercel (frontend), Render/Railway (backend) |
| CI/CD | GitHub Actions |

---

## Data Sources to Explore

- U.S. Census Bureau (population, demographics, income)
- Bureau of Labor Statistics (employment by industry, wages)
- Zillow / Redfin (commercial real estate costs)
- NCSES / NSF (R&D expenditure by region)
- University locations (proximity to talent pipelines)
- NIH funding by state (pharma relevance)
- State tax incentive databases
- Infrastructure scores (broadband, transit)

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| Insufficient labeled data for supervised learning | Use unsupervised clustering (DBSCAN, k-means) to generate pseudo-labels, or pivot to a scoring/ranking approach |
| Neural network doesn't outperform simple models | Keep baseline model as production fallback — prioritize result quality over model complexity |
| Data collection bottleneck | Start with a focused geography (e.g., top 200 U.S. metro areas) and expand later |
| Integration delays | Define API contract early (Week 4); frontend and backend can develop in parallel with mocked data |
| Scope creep | Freeze feature set after Week 7; Weeks 8–9 are polish only |
