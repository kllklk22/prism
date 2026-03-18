# prism

> **parallel exploratory data analysis engine — upload a csv, watch julia tear it apart across every cpu core simultaneously.**

a full-stack data analysis tool built for scientists and researchers who are tired of waiting on python to loop through columns one at a time. prism dispatches every analysis task — distributions, correlations, outliers, pca, missing value profiling — to a separate julia worker process. they all run at the same time. results stream back to a live dashboard as each worker finishes.

written by me.

---

## what it does

- **parallel execution** — spawns `N_CORES - 1` julia workers on startup. every analysis task runs on a different worker simultaneously, not sequentially.
- **summary statistics** — mean, std, min/max, quartiles, skewness, kurtosis per numeric column
- **distribution histograms** — 30-bin adaptive histograms, one per numeric column
- **correlation matrix** — pearson correlation heatmap across all numeric columns
- **pca** — power-iteration based, two components, variance explained, scatter plot
- **outlier detection** — iqr method and z-score method (|z|>3), percent flagged per column
- **missing value profile** — counts, percentages, visual bar chart per column
- **live dashboard** — polls job status every 1.5s, panels render as workers finish

---

## why julia

julia's `Distributed` standard library makes true multiprocessing trivial. `@spawnat :any` sends a closure to any available worker and returns a `Future` immediately. all futures are created before any are awaited — meaning all workers start simultaneously. fetching results blocks until each finishes, but the actual computation overlapped.

a dataset with 20 numeric columns dispatches 60+ concurrent analysis tasks (summary + histogram + outlier per column, plus correlation and pca). on an 8-core machine that's all running at the same time. python is doing it in a for loop right now. sequentially. sadly.

---

## stack

- **backend** — julia 1.9+, `HTTP.jl`, `Distributed`, `CSV.jl`, `DataFrames.jl`, `JSON3.jl`, `Statistics`, `StatsBase`
- **frontend** — single html file, `Chart.js`, no framework, no build step
- **architecture** — julia http server serves both the api and the static frontend

---

## requirements

- julia 1.9 or later — https://julialang.org/downloads
- that's it. julia manages its own packages.

---

## setup

**step 1 — clone and install dependencies**

```bash
git clone https://github.com/YOURUSERNAME/prism.git
cd prism
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

**step 2 — run the server**

```bash
julia --project=. server.jl
```

the server will print how many workers it spawned and what port it's on. default is `8080`.

```
██████╗ ██████╗ ██╗███████╗███╗   ███╗
...
workers:  7 (8 cores detected)
server:   http://localhost:8080
```

**step 3 — open the dashboard**

go to `http://localhost:8080` in your browser. drag and drop a csv. watch the panels populate.

---

## project structure

```
prism/
├── server.jl          ← http server, job queue, parallel dispatch
├── analysis.jl        ← analysis functions loaded onto every worker
├── public/
│   └── index.html     ← full dashboard frontend, no build step needed
├── Project.toml       ← julia package dependencies
└── README.md          ← you're here
```

---

## api

the server exposes three endpoints. the frontend uses all of them.

- `POST /upload` — multipart csv upload, returns job id immediately
- `GET  /status/:id` — job status + full results when complete
- `GET  /jobs` — list all jobs (useful for debugging)

job lifecycle: `pending → running → complete` (or `failed` if something explodes)

---

## configuration

environment variables:

- `PRISM_PORT` — server port, default `8080`

worker count is automatically set to `Sys.CPU_THREADS - 1`. one core runs the http server, everything else runs analysis. if you want to override this, change `N_WORKERS` in `server.jl`. don't set it higher than your core count. you know why.

---

## extending it

adding a new analysis type:

1. write a function in `analysis.jl` — it must be self-contained (workers don't share memory)
2. add a `@spawnat :any your_function(args)` line in `run_analyses_parallel` in `server.jl`
3. `fetch()` the future and add results to the response dict
4. render the result in `index.html`

that's the whole pattern. everything else is plumbing.

---

## notes

- the power iteration pca is intentional — it avoids a full eigendecomposition and is fast enough for dashboard purposes. if you need exact eigenvectors, swap in `MultivariateStats.jl`'s `fit(PCA, ...)`.
- correlation matrix uses pearson. spearman would require rank transformation per column. that's in the roadmap.
- missing value detection catches: empty string, `NA`, `na`, `NaN`, `null`, `NULL`, `None`, `N/A`. if your dataset uses something else, edit `compute_missing_profile` in `analysis.jl`.
- the server holds jobs in memory. restart = jobs gone. this is a feature for a privacy-sensitive research context. if you need persistence, add sqlite.

---

## github description

> parallel eda engine built on julia's distributed computing. upload a csv, get distributions, correlations, pca, outlier detection and missing value profiling — all running simultaneously across cpu cores.

---

*written entirely by dande. yes it runs. no i won't debug your environment.*
