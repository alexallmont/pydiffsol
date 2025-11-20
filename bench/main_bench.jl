include("diffeq_models.jl")
using BenchmarkTools
using DataFrames
using CSV

function main_bench(torun)

    # Create DataFrame to store timing results
    results = DataFrame(problem=String[], ngroups=Int[], tol=Float64[],
                        n_runs=Int[],
                        diffeq_bdf_time=Float64[],
                        diffeq_bdf_setup_time=Float64[],
                        diffeq_kencarp3_time=Float64[],
                        diffeq_kencarp3_setup_time=Float64[],
                        diffeq_tsit5_setup_time=Float64[],
                        diffeq_tsit5_time=Float64[],
                        diffeq_tr_bdf2_time=Float64[],
                        diffeq_tr_bdf2_setup_time=Float64[])

    for run in torun
        (ng, tol, problem) = run
        println("ngroups: ", ng)
        println("tol: ", tol)
        println("problem: ", problem)
        is_stiff = problem == "robertson_ode"
        n = 500 ÷ (floor(0.01 * ng) + 1)
        diffeq_bdf_setup_time = @elapsed model_bdf = setup(ng, tol, "bdf", problem)
        diffeq_kencarp3_setup_time = @elapsed model_kencarp3 = setup(ng, tol, "kencarp3", problem)
        diffeq_tr_bdf2_setup_time = @elapsed model_tr_bdf2 = setup(ng, tol, "tr_bdf2", problem)

        if !is_stiff
            diffeq_tsit5_setup_time = @elapsed model_tsit5 = setup(ng, tol, "tsit5", problem)
        else
            diffeq_tsit5_setup_time = 0.0
        end

        # run all the benchmarks so that compilation time is excluded from timing, add to setup times
        diffeq_bdf_setup_time += @elapsed bench(model_bdf)
        diffeq_kencarp3_setup_time += @elapsed bench(model_kencarp3)
        diffeq_tr_bdf2_setup_time += @elapsed bench(model_tr_bdf2)
        if !is_stiff
            diffeq_tsit5_setup_time += @elapsed bench(model_tsit5)
        end


        t = @benchmark bench($model_bdf) samples=n
        # returns times in nanoseconds, convert to seconds
        diffeq_bdf_time = mean(t.times) / 1e9
        t = @benchmark bench($model_kencarp3) samples=n
        diffeq_kencarp3_time = mean(t.times) / 1e9
        t = @benchmark bench($model_tr_bdf2) samples=n
        diffeq_tr_bdf2_time = mean(t.times) / 1e9
        if !is_stiff
            t = @benchmark bench($model_tsit5) samples=n
            diffeq_tsit5_time = mean(t.times) / 1e9
        else
            diffeq_tsit5_time = 0.0
        end

        println("DiffEq BDF setup time: ", diffeq_bdf_setup_time)
        println("DiffEq KenCarp3 setup time: ", diffeq_kencarp3_setup_time)
        println("DiffEq TR-BDF2 setup time: ", diffeq_tr_bdf2_setup_time)
        println("DiffEq BDF time: ", diffeq_bdf_time)
        println("DiffEq KenCarp3 time: ", diffeq_kencarp3_time)
        println("DiffEq TR-BDF2 time: ", diffeq_tr_bdf2_time)
        println("DiffEq Tsit5 setup time: ", diffeq_tsit5_setup_time)
        println("DiffEq Tsit5 time: ", diffeq_tsit5_time)

        # Prepare result row
        result_row = (problem=problem, ngroups=ng, tol=tol, n_runs=n,
                      diffeq_bdf_time=diffeq_bdf_time,
                      diffeq_bdf_setup_time=diffeq_bdf_setup_time,
                      diffeq_kencarp3_time=diffeq_kencarp3_time,
                      diffeq_kencarp3_setup_time=diffeq_kencarp3_setup_time,
                      diffeq_tsit5_setup_time=diffeq_tsit5_setup_time,
                      diffeq_tsit5_time=diffeq_tsit5_time,
                      diffeq_tr_bdf2_time=diffeq_tr_bdf2_time,
                      diffeq_tr_bdf2_setup_time=diffeq_tr_bdf2_setup_time)
        push!(results, result_row)
    end
    println(results)
    CSV.write("benchmark_results_robertson_ode_jl.csv", results)
end
                          

main_bench(vcat([
    (ng, tol, "lotka_volterra_ode") for ng in [1] for tol in [1e-2, 1e-4, 1e-6, 1e-8]
], [
    (ng, tol, "robertson_ode") for ng in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] for tol in [1e-8]
]))