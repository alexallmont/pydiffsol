import timeit
import numpy as np
import pandas as pd
import time
from diffrax_models import setup as diffrax_setup, bench as diffrax_bench
from casadi_models import setup as casadi_setup, bench as casadi_bench
from diffsol_models import setup as diffsol_setup, bench as diffsol_bench


def bench(torun):
    results = []
    for run in torun:
        (ng, tol, problem) = run

        is_stiff = problem == "robertson_ode"

        t0 = time.perf_counter()
        diffrax_kvaerno5_model = diffrax_setup(ngroups=ng, tol=tol, method='kvaerno5', problem=problem)
        t1 = time.perf_counter()
        diffrax_kvaerno5_setup_time = t1 - t0

        def diffrax_kvaerno5():
            return diffrax_bench(diffrax_kvaerno5_model)

        t0 = time.perf_counter()
        diffrax_tsit5_model = diffrax_setup(ngroups=ng, tol=tol, method='tsit5', problem=problem)
        t1 = time.perf_counter()
        diffrax_tsit5_setup_time = t1 - t0

        def diffrax_tsit5():
            return diffrax_bench(diffrax_tsit5_model)

        t0 = time.perf_counter()
        casadi_model = casadi_setup(ngroups=ng, tol=tol, problem=problem)
        t1 = time.perf_counter()
        casadi_setup_time = t1 - t0

        def casadi():
            return casadi_bench(casadi_model)

        t0 = time.perf_counter()
        diffsol_bdf_model = diffsol_setup(ngroups=ng, tol=tol, method="bdf", problem=problem)
        t1 = time.perf_counter()
        diffsol_setup_time = t1 - t0

        def diffsol_bdf():
            return diffsol_bench(diffsol_bdf_model)

        diffsol_esdirk34_model = diffsol_setup(
            ngroups=ng, tol=tol, method="esdirk34", problem=problem
        )

        def diffsol_esdirk34():
            return diffsol_bench(diffsol_esdirk34_model)

        diffsol_tr_bdf2_model = diffsol_setup(
            ngroups=ng, tol=tol, method="tr_bdf2", problem=problem
        )

        def diffsol_tr_bdf2():
            return diffsol_bench(diffsol_tr_bdf2_model)

        diffsol_tsit5_model = diffsol_setup(
            ngroups=ng, tol=tol, method="tsit5", problem=problem
        )

        def diffsol_tsit5():
            return diffsol_bench(diffsol_tsit5_model)

        run_diffrax = ng <= 100

        # check that output is same
        t0 = time.perf_counter()
        y_casadi = casadi()
        t1 = time.perf_counter()
        casadi_setup_time += t1 - t0
        y_casadi = np.array(y_casadi).flatten()
        t0 = time.perf_counter()
        y_diffsol_bdf = diffsol_bdf()
        t1 = time.perf_counter()
        diffsol_setup_time += t1 - t0
        y_diffsol_esdirk34 = diffsol_esdirk34()
        y_diffsol_tr_bdf2 = diffsol_tr_bdf2()

        check_tol = 1e3 * tol

        np.testing.assert_allclose(
            y_casadi, y_diffsol_bdf, rtol=check_tol, atol=check_tol
        )
        np.testing.assert_allclose(
            y_casadi, y_diffsol_esdirk34, rtol=check_tol, atol=check_tol
        )
        np.testing.assert_allclose(
            y_casadi, y_diffsol_tr_bdf2, rtol=check_tol, atol=check_tol
        )
        if run_diffrax:
            t0 = time.perf_counter()
            y_diffrax = diffrax_kvaerno5()
            t1 = time.perf_counter()
            diffrax_kvaerno5_setup_time += t1 - t0
            np.testing.assert_allclose(
                y_casadi, y_diffrax, rtol=check_tol, atol=check_tol
            )
        if not is_stiff:
            y_diffsol = diffsol_tsit5()
            np.testing.assert_allclose(
                y_casadi, y_diffsol, rtol=check_tol, atol=check_tol
            )
            if run_diffrax:
                t0 = time.perf_counter()
                y_diffrax = diffrax_tsit5()
                t1 = time.perf_counter()
                diffrax_tsit5_setup_time += t1 - t0
                np.testing.assert_allclose(
                    y_casadi, y_diffrax, rtol=check_tol, atol=check_tol
                )

        n = 500 // (int(0.01 * ng) + 1)
        print("ngroups: ", ng)
        print("tol: ", tol)
        print("n: ", n)
        print("problem: ", problem)

        casadi_time = timeit.timeit(casadi, number=n) / n
        print("Casadi setup time: ", casadi_setup_time)
        print("Casadi time: ", casadi_time)
        diffsol_bdf_time = timeit.timeit(diffsol_bdf, number=n) / n
        print("Diffsol BDF setup time: ", diffsol_setup_time)
        print("Diffsol BDF time: ", diffsol_bdf_time)
        diffsol_esdirk34_time = timeit.timeit(diffsol_esdirk34, number=n) / n
        print("Diffsol ESDIRK34 time: ", diffsol_esdirk34_time)
        diffsol_tr_bdf2_time = timeit.timeit(diffsol_tr_bdf2, number=n) / n
        print("Diffsol TR-BDF2 time: ", diffsol_tr_bdf2_time)
        print("Speedup over casadi: ", casadi_time / diffsol_bdf_time)

        # Prepare result row
        result_row = {
            "problem": problem,
            "ngroups": ng,
            "tol": tol,
            "n_runs": n,
            "casadi_setup_time": casadi_setup_time,
            "casadi_time": casadi_time,
            "diffsol_setup_time": diffsol_setup_time,
            "diffsol_bdf_time": diffsol_bdf_time,
            "diffsol_esdirk34_time": diffsol_esdirk34_time,
            "diffsol_tr_bdf2_time": diffsol_tr_bdf2_time,
            "diffsol_tsit5_time": None,
            "speedup_casadi_vs_bdf": casadi_time / diffsol_bdf_time,
            "diffrax_kvaerno5_setup_time": None,
            "diffrax_kvaerno5_time": None,
            "diffrax_tsit5_setup_time": None,
            "diffrax_tsit5_time": None,
            "speedup_diffrax_vs_bdf": None,
        }

        if not is_stiff:
            diffsol_tsit5_time = timeit.timeit(diffsol_tsit5, number=n) / n
            print("Diffsol tsit5 time: ", diffsol_tsit5_time)
            result_row["diffsol_tsit5_time"] = diffsol_tsit5_time
            if run_diffrax:
                diffrax_tsit5_time = timeit.timeit(diffrax_tsit5, number=n) / n
                print("Diffrax tsit5 setup time: ", diffrax_tsit5_setup_time)
                print("Diffrax tsit5 time: ", diffrax_tsit5_time)
                result_row["diffrax_tsit5_setup_time"] = diffrax_tsit5_setup_time
                result_row["diffrax_tsit5_time"] = diffrax_tsit5_time

        if run_diffrax:
            diffrax_kvaerno5_time = timeit.timeit(diffrax_kvaerno5, number=n) / n
            print("Diffrax kvaerno5 setup time: ", diffrax_kvaerno5_setup_time)
            print("Speedup over diffrax: ", diffrax_kvaerno5_time / diffsol_bdf_time)
            result_row["diffrax_kvaerno5_setup_time"] = diffrax_kvaerno5_setup_time
            result_row["diffrax_kvaerno5_time"] = diffrax_kvaerno5_time


        # Add result to list
        results.append(result_row)

    # Create DataFrame from results
    df_results = pd.DataFrame(results)

    # Display the results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    print(df_results.to_string(index=False))

    # Save results to CSV
    csv_filename = f"benchmark_results_robertson_ode.csv"
    df_results.to_csv(csv_filename, index=False)
    print(f"\nResults saved to: {csv_filename}")


# Smoke test docs code
def test_robertson_ode_bench():
    bench([
        (ng, tol, "robertson_ode") for ng in [1, 2] for tol in [1e-8]
    ])


if __name__ == "__main__":
    bench(
        torun=[
            (ng, tol, "lotka_volterra_ode")
            for ng in [1]
            for tol in [1e-2, 1e-4, 1e-6, 1e-8]
        ]
        + [
            (ng, tol, "robertson_ode")
            for ng in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
            for tol in [1e-8]
        ]
    )
