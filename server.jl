# server.jl
# dande. prism. parallel eda engine.
# adds one worker per available cpu core on startup.
# if your machine has 2 cores this will feel slow. get a better machine.
# if your machine has 16 cores this will feel unreasonably fast. you're welcome.

using Distributed

# spin up workers before loading anything else
# Sys.CPU_THREADS - 1 because one core runs the server, the rest do your math
const N_WORKERS = max(1, Sys.CPU_THREADS - 1)
addprocs(N_WORKERS)

# load analysis functions on EVERY worker
# @everywhere means exactly what it sounds like. everywhere.
@everywhere include(joinpath(@__DIR__, "analysis.jl"))

using HTTP
using JSON3
using CSV
using DataFrames
using Dates
using Statistics

# ─── job state ────────────────────────────────────────────────
# not a database. if you restart the server the jobs are gone.
# this is an analysis tool, not a data warehouse. cope.

@enum JobStatus pending running complete failed

mutable struct Job
    id        :: String
    status    :: JobStatus
    created   :: DateTime
    filename  :: String
    n_rows    :: Int
    n_cols    :: Int
    col_names :: Vector{String}
    results   :: Dict{String, Any}
    error     :: String
end

Job(id, fname) = Job(id, pending, now(), fname, 0, 0, String[], Dict{String,Any}(), "")

const JOBS   = Dict{String, Job}()
const JOB_LOCK = ReentrantLock()

function new_job_id()
    hex = string(rand(UInt64), base=16, pad=16)
    return hex[1:8] * "-" * hex[9:12] * "-" * hex[13:16]
end

# ─── csv parsing ──────────────────────────────────────────────
function parse_csv_bytes(data::Vector{UInt8})
    # CSV.jl wants an io. give it an io.
    io = IOBuffer(data)
    df = CSV.read(io, DataFrame; missingstring=["", "NA", "na", "NaN", "null", "NULL", "None", "N/A"])
    return df
end

function extract_numeric_cols(df::DataFrame)
    numeric = Dict{String, Vector{Float64}}()
    for col in names(df)
        T = eltype(df[!, col])
        if T <: Union{Missing, Float64} || T <: Union{Missing, Float32} ||
           T <: Union{Missing, Int64}   || T <: Union{Missing, Int32}   ||
           T <: Union{Missing, Number}
            vals = Float64[ismissing(v) ? NaN : Float64(v) for v in df[!, col]]
            clean = filter(!isnan, vals)
            length(clean) >= 2 && (numeric[string(col)] = clean)
        end
    end
    return numeric
end

function extract_raw_cols(df::DataFrame)
    raw = Dict{String, Vector{String}}()
    for col in names(df)
        raw[string(col)] = [ismissing(v) ? "" : string(v) for v in df[!, col]]
    end
    return raw
end

# ─── parallel analysis dispatch ───────────────────────────────
# this is the whole point of the project.
# each analysis task gets spawned onto a different worker.
# they all run at the same time. simultaneously. in parallel.
# python is doing them one at a time right now, sequentially, sadly.
function run_analyses_parallel(job::Job, df::DataFrame)
    lock(JOB_LOCK) do
        job.status  = running
        job.n_rows  = nrow(df)
        job.n_cols  = ncol(df)
        job.col_names = names(df)
    end

    numeric_cols = extract_numeric_cols(df)
    raw_cols     = extract_raw_cols(df)
    col_list     = collect(keys(numeric_cols))

    # ── spawn all tasks onto distributed workers ──────────────
    # each @spawnat picks a worker and sends it a closure.
    # the futures are all created instantly — then we wait on all of them.
    futures = Dict{String, Any}()

    # missing value profile — runs on worker 1 (or whatever's free)
    futures["missing"] = @spawnat :any compute_missing_profile(raw_cols)

    # per-column summaries and histograms — one future per column
    # if you have 20 columns and 16 workers, 16 columns run simultaneously
    # then the remaining 4. still faster than python's sequential loop.
    summary_futures = Dict{String, Any}()
    hist_futures    = Dict{String, Any}()
    outlier_futures = Dict{String, Any}()

    for (name, col) in numeric_cols
        summary_futures[name] = @spawnat :any compute_summary(col)
        hist_futures[name]    = @spawnat :any compute_histogram(col)
        outlier_futures[name] = @spawnat :any compute_outliers(col)
    end

    # correlation matrix — needs all columns, single task
    if length(col_list) >= 2
        futures["correlations"] = @spawnat :any compute_correlations(numeric_cols)
    end

    # pca — needs all columns, single task
    if length(col_list) >= 2
        futures["pca"] = @spawnat :any compute_pca(numeric_cols)
    end

    # ── collect results as they finish ────────────────────────
    results = Dict{String, Any}()

    # collect missing
    try
        results["missing"] = fetch(futures["missing"])
    catch e
        results["missing"] = Dict("error" => string(e))
    end

    # collect per-column results
    summaries = Dict{String, Any}()
    histograms = Dict{String, Any}()
    outliers   = Dict{String, Any}()

    for name in col_list
        try summaries[name] = fetch(summary_futures[name])
        catch e; summaries[name] = Dict("error" => string(e)); end

        try histograms[name] = fetch(hist_futures[name])
        catch e; histograms[name] = Dict("error" => string(e)); end

        try outliers[name] = fetch(outlier_futures[name])
        catch e; outliers[name] = Dict("error" => string(e)); end
    end

    results["summaries"]  = summaries
    results["histograms"] = histograms
    results["outliers"]   = outliers

    # collect correlation + pca
    if haskey(futures, "correlations")
        try results["correlations"] = fetch(futures["correlations"])
        catch e; results["correlations"] = Dict("error" => string(e)); end
    end

    if haskey(futures, "pca")
        try results["pca"] = fetch(futures["pca"])
        catch e; results["pca"] = Dict("error" => string(e)); end
    end

    # metadata
    results["meta"] = Dict(
        "filename"     => job.filename,
        "n_rows"       => job.n_rows,
        "n_cols"       => job.n_cols,
        "col_names"    => job.col_names,
        "numeric_cols" => col_list,
        "n_workers"    => N_WORKERS,
        "computed_at"  => string(now()),
    )

    lock(JOB_LOCK) do
        job.results = results
        job.status  = complete
    end
end

# ─── http handlers ────────────────────────────────────────────

# serve the frontend
function handle_root(req::HTTP.Request)
    html_path = joinpath(@__DIR__, "public", "index.html")
    isfile(html_path) || return HTTP.Response(404, "index.html not found")
    return HTTP.Response(200,
        ["Content-Type" => "text/html; charset=utf-8"],
        read(html_path)
    )
end

# upload csv, create job, dispatch workers, return job id immediately
function handle_upload(req::HTTP.Request)
    # multipart parse — find the csv part
    body = req.body
    content_type = HTTP.header(req, "Content-Type", "")

    if !occursin("multipart/form-data", content_type)
        return json_response(400, Dict("error" => "expected multipart/form-data"))
    end

    boundary = match(r"boundary=(.+)", content_type)
    boundary === nothing && return json_response(400, Dict("error" => "no boundary found"))
    bnd = "--" * strip(boundary.captures[1])

    body_str  = String(body)
    parts     = split(body_str, bnd)
    csv_data  = UInt8[]
    filename  = "upload.csv"

    for part in parts
        if occursin("Content-Disposition", part) && occursin("filename=", part)
            fn_match = match(r"filename=\"([^\"]+)\"", part)
            fn_match !== nothing && (filename = fn_match.captures[1])
            # data starts after double CRLF
            data_start = findfirst("\r\n\r\n", part)
            if data_start !== nothing
                raw = part[(data_start.stop+1):end]
                # strip trailing boundary marker noise
                raw = replace(raw, r"\r\n--$" => "")
                csv_data = Vector{UInt8}(raw)
            end
        end
    end

    isempty(csv_data) && return json_response(400, Dict("error" => "no csv data found in upload"))

    # parse
    df = try
        parse_csv_bytes(csv_data)
    catch e
        return json_response(400, Dict("error" => "csv parse failed: " * string(e)))
    end

    nrow(df) == 0 && return json_response(400, Dict("error" => "empty dataframe"))

    # create job
    job_id = new_job_id()
    job    = Job(job_id, filename)
    lock(JOB_LOCK) do
        JOBS[job_id] = job
    end

    # dispatch in background — don't block the http response
    @async begin
        try
            run_analyses_parallel(job, df)
        catch e
            lock(JOB_LOCK) do
                job.status = failed
                job.error  = string(e)
            end
        end
    end

    return json_response(202, Dict(
        "job_id"   => job_id,
        "filename" => filename,
        "n_rows"   => nrow(df),
        "n_cols"   => ncol(df),
        "status"   => "running",
    ))
end

# poll job status and results
function handle_status(req::HTTP.Request)
    # extract job id from path: /status/:id
    m = match(r"/status/([^/]+)$", req.target)
    m === nothing && return json_response(400, Dict("error" => "missing job id"))
    job_id = m.captures[1]

    job = lock(JOB_LOCK) do
        get(JOBS, job_id, nothing)
    end

    job === nothing && return json_response(404, Dict("error" => "job not found"))

    status_str = job.status == pending  ? "pending"  :
                 job.status == running  ? "running"  :
                 job.status == complete ? "complete" : "failed"

    response = Dict(
        "job_id"   => job_id,
        "status"   => status_str,
        "filename" => job.filename,
        "n_rows"   => job.n_rows,
        "n_cols"   => job.n_cols,
    )

    if job.status == complete
        response["results"] = job.results
    elseif job.status == failed
        response["error"] = job.error
    end

    return json_response(200, response)
end

# list all jobs — useful for debugging, not much else
function handle_jobs(req::HTTP.Request)
    job_list = lock(JOB_LOCK) do
        [Dict(
            "id"       => id,
            "status"   => string(j.status),
            "filename" => j.filename,
            "n_rows"   => j.n_rows,
            "created"  => string(j.created),
        ) for (id, j) in JOBS]
    end
    return json_response(200, Dict("jobs" => job_list))
end

function json_response(status::Int, body)
    return HTTP.Response(
        status,
        ["Content-Type" => "application/json", "Access-Control-Allow-Origin" => "*"],
        JSON3.write(body)
    )
end

# ─── router ───────────────────────────────────────────────────
function router(req::HTTP.Request)
    method = req.method
    target = req.target

    method == "GET"  && target == "/"             && return handle_root(req)
    method == "POST" && target == "/upload"        && return handle_upload(req)
    method == "GET"  && startswith(target,"/status") && return handle_status(req)
    method == "GET"  && target == "/jobs"          && return handle_jobs(req)

    # cors preflight — browser will ask before uploading
    method == "OPTIONS" && return HTTP.Response(204,
        ["Access-Control-Allow-Origin"  => "*",
         "Access-Control-Allow-Methods" => "GET, POST, OPTIONS",
         "Access-Control-Allow-Headers" => "Content-Type"])

    return HTTP.Response(404, "not found")
end

# ─── startup ──────────────────────────────────────────────────
function main()
    port = parse(Int, get(ENV, "PRISM_PORT", "8080"))

    println("""
    ██████╗ ██████╗ ██╗███████╗███╗   ███╗
    ██╔══██╗██╔══██╗██║██╔════╝████╗ ████║
    ██████╔╝██████╔╝██║███████╗██╔████╔██║
    ██╔═══╝ ██╔══██╗██║╚════██║██║╚██╔╝██║
    ██║     ██║  ██║██║███████║██║ ╚═╝ ██║
    ╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝
    parallel eda engine — by dande
    """)

    println("workers:  $N_WORKERS ($(Sys.CPU_THREADS) cores detected)")
    println("server:   http://localhost:$port")
    println("upload:   POST /upload")
    println("status:   GET  /status/:id\n")

    HTTP.serve(router, "0.0.0.0", port)
end

main()
