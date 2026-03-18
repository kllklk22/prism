# analysis.jl
# dande. these run on distributed workers.
# each function is self-contained — that's not laziness, that's correctness.
# workers don't share memory. stop trying to share memory.

using Statistics
using StatsBase

# ─── summary statistics ───────────────────────────────────────
# the basics. still useful. stop pretending you don't need them.
function compute_summary(data::Vector{Float64})
    n = length(data)
    n == 0 && return Dict("error" => "empty column")

    sorted = sort(data)
    q1  = quantile(data, 0.25)
    med = median(data)
    q3  = quantile(data, 0.75)
    iqr = q3 - q1

    return Dict(
        "type"     => "summary",
        "n"        => n,
        "mean"     => mean(data),
        "std"      => std(data),
        "min"      => minimum(data),
        "q1"       => q1,
        "median"   => med,
        "q3"       => q3,
        "max"      => maximum(data),
        "iqr"      => iqr,
        "skewness" => _skewness(data),
        "kurtosis" => _kurtosis(data),
    )
end

function _skewness(x::Vector{Float64})
    n = length(x)
    n < 3 && return 0.0
    μ = mean(x)
    σ = std(x)
    σ == 0.0 && return 0.0
    sum(((xi - μ) / σ)^3 for xi in x) * n / ((n-1) * (n-2))
end

function _kurtosis(x::Vector{Float64})
    n = length(x)
    n < 4 && return 0.0
    μ = mean(x)
    σ = std(x)
    σ == 0.0 && return 0.0
    (sum(((xi - μ) / σ)^4 for xi in x) * n*(n+1) / ((n-1)*(n-2)*(n-3))) -
    (3*(n-1)^2 / ((n-2)*(n-3)))
end

# ─── histogram ────────────────────────────────────────────────
function compute_histogram(data::Vector{Float64}; bins::Int=30)
    length(data) == 0 && return Dict("error" => "empty")

    lo, hi = minimum(data), maximum(data)
    lo == hi && return Dict(
        "type"   => "histogram",
        "bins"   => [lo],
        "counts" => [length(data)],
        "edges"  => [lo, lo+1.0],
    )

    edges  = range(lo, hi, length=bins+1)
    counts = zeros(Int, bins)

    for v in data
        idx = min(searchsortedlast(collect(edges), v), bins)
        idx > 0 && (counts[idx] += 1)
    end

    return Dict(
        "type"   => "histogram",
        "bins"   => [(edges[i] + edges[i+1]) / 2 for i in 1:bins],
        "counts" => counts,
        "edges"  => collect(edges),
    )
end

# ─── outlier detection ────────────────────────────────────────
# iqr method + z-score method. returns indices and values.
# yes both. they disagree sometimes. that's your problem to interpret.
function compute_outliers(data::Vector{Float64})
    length(data) == 0 && return Dict("error" => "empty")

    # iqr method
    q1 = quantile(data, 0.25)
    q3 = quantile(data, 0.75)
    iqr_val = q3 - q1
    lower = q1 - 1.5 * iqr_val
    upper = q3 + 1.5 * iqr_val
    iqr_outliers = [(i, v) for (i, v) in enumerate(data) if v < lower || v > upper]

    # z-score method (|z| > 3)
    μ = mean(data)
    σ = std(data)
    z_outliers = σ > 0 ?
        [(i, v, (v - μ) / σ) for (i, v) in enumerate(data) if abs((v - μ) / σ) > 3.0] :
        []

    return Dict(
        "type"           => "outliers",
        "iqr_lower"      => lower,
        "iqr_upper"      => upper,
        "iqr_count"      => length(iqr_outliers),
        "iqr_indices"    => first.(iqr_outliers),
        "iqr_values"     => last.(iqr_outliers),
        "zscore_count"   => length(z_outliers),
        "zscore_indices" => [t[1] for t in z_outliers],
        "zscore_values"  => [t[2] for t in z_outliers],
        "zscore_zscores" => [t[3] for t in z_outliers],
        "pct_flagged"    => round(length(iqr_outliers) / length(data) * 100, digits=2),
    )
end

# ─── correlation matrix ───────────────────────────────────────
# pearson. all numeric columns against each other.
# slow in python. not slow here. that's the whole point of this project.
function compute_correlations(cols::Dict{String, Vector{Float64}})
    names = collect(keys(cols))
    n = length(names)
    n < 2 && return Dict("type" => "correlations", "names" => names, "matrix" => [[1.0]])

    matrix = zeros(Float64, n, n)
    for i in 1:n, j in 1:n
        if i == j
            matrix[i, j] = 1.0
        elseif j > i
            r = _pearson(cols[names[i]], cols[names[j]])
            matrix[i, j] = r
            matrix[j, i] = r
        end
    end

    return Dict(
        "type"   => "correlations",
        "names"  => names,
        "matrix" => [matrix[i, :] for i in 1:n],
    )
end

function _pearson(x::Vector{Float64}, y::Vector{Float64})
    n = min(length(x), length(y))
    n < 2 && return 0.0
    x, y = x[1:n], y[1:n]
    μx, μy = mean(x), mean(y)
    num = sum((x .- μx) .* (y .- μy))
    den = sqrt(sum((x .- μx).^2) * sum((y .- μy).^2))
    den == 0.0 ? 0.0 : num / den
end

# ─── missing value profile ────────────────────────────────────
function compute_missing_profile(raw_cols::Dict{String, Vector{String}})
    result = Dict{String, Any}()
    total_rows = maximum(length(v) for v in values(raw_cols); init=0)

    col_profiles = Dict{String, Any}()
    for (name, vals) in raw_cols
        missing_count = count(v -> strip(v) == "" || lowercase(strip(v)) in ("na", "nan", "null", "none", "n/a"), vals)
        col_profiles[name] = Dict(
            "missing"     => missing_count,
            "present"     => length(vals) - missing_count,
            "pct_missing" => round(missing_count / max(length(vals), 1) * 100, digits=2),
        )
    end

    return Dict(
        "type"        => "missing",
        "total_rows"  => total_rows,
        "total_cols"  => length(raw_cols),
        "columns"     => col_profiles,
        "total_cells" => total_rows * length(raw_cols),
        "total_missing" => sum(p["missing"] for p in values(col_profiles)),
    )
end

# ─── pca ──────────────────────────────────────────────────────
# two components. enough for visualization. if you need 40 components
# you need a therapist, not a dashboard.
function compute_pca(cols::Dict{String, Vector{Float64}})
    names  = collect(keys(cols))
    length(names) < 2 && return Dict("error" => "need at least 2 numeric columns for pca")

    n_rows = minimum(length(v) for v in values(cols))
    n_rows < 3 && return Dict("error" => "need at least 3 rows for pca")

    # build matrix: rows = observations, cols = features
    X = zeros(Float64, n_rows, length(names))
    for (j, name) in enumerate(names)
        col = cols[name][1:n_rows]
        μ = mean(col)
        σ = std(col)
        X[:, j] = σ > 0 ? (col .- μ) ./ σ : col .- μ
    end

    # covariance matrix
    C = (X' * X) ./ (n_rows - 1)

    # power iteration for top 2 eigenvectors — good enough
    pc1 = _power_iter(C, 50)
    deflated = C - (pc1 * pc1') .* (pc1' * C * pc1)
    pc2 = _power_iter(deflated, 50)

    # project
    scores1 = X * pc1
    scores2 = X * pc2

    # variance explained
    total_var = sum(diag(C))
    var1 = abs(pc1' * C * pc1)
    var2 = abs(pc2' * C * pc2)

    return Dict(
        "type"             => "pca",
        "scores_pc1"       => scores1,
        "scores_pc2"       => scores2,
        "loadings_pc1"     => Dict(names[i] => pc1[i] for i in eachindex(names)),
        "loadings_pc2"     => Dict(names[i] => pc2[i] for i in eachindex(names)),
        "var_explained_pc1"=> round(var1 / total_var * 100, digits=2),
        "var_explained_pc2"=> round(var2 / total_var * 100, digits=2),
        "feature_names"    => names,
        "n_samples"        => n_rows,
    )
end

function _power_iter(A::Matrix{Float64}, iters::Int)
    n = size(A, 1)
    v = randn(n)
    v ./= norm(v)
    for _ in 1:iters
        v = A * v
        nv = norm(v)
        nv > 0 && (v ./= nv)
    end
    return v
end

function norm(v::Vector{Float64})
    sqrt(sum(v .^ 2))
end

function diag(A::Matrix{Float64})
    [A[i,i] for i in 1:min(size(A)...)]
end
