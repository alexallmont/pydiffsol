import timeit
import numpy as np
import pandas as pd
from .diffrax_robertson_ode import setup as diffrax_setup, bench as diffrax_bench
from .casadi_robertson_ode import setup as casadi_setup, bench as casadi_bench
from .diffsol_robertson_ode import setup as diffsol_setup, bench as diffsol_bench


ngroups = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
tols = [1e-8]
t_final = 1e10

# Create DataFrame to store timing results
results = []


diffrax_fns = []
for ng in ngroups:
    for tol in tols:

        diffrax_model = diffrax_setup(ngroups=ng)

        def diffrax():
            return diffrax_bench(diffrax_model, ngroups=ng, t_final=t_final, tol=tol)

        casadi_model = casadi_setup(ngroups=ng, tol=tol, t_final=t_final)

        def casadi():
            return casadi_bench(casadi_model, ng)

        diffsol_bdf_model = diffsol_setup(ngroups=ng, tol=tol, method="bdf")

        def diffsol_bdf():
            return diffsol_bench(diffsol_bdf_model, t_final)
        
        diffsol_esdirk34_model = diffsol_setup(ngroups=ng, tol=tol, method="esdirk34")

        def diffsol_esdirk34():
            return diffsol_bench(diffsol_esdirk34_model, t_final)
        
        
        run_diffrax = ng <= 100
        
        # check that output is same
        y_casadi = casadi()
        y_casadi = np.array(y_casadi).flatten()
        y_diffsol_bdf = diffsol_bdf()
        y_diffsol_esdirk34 = diffsol_esdirk34()
        check_tol = 20*tol
        if run_diffrax:
            y_diffrax = diffrax()
            np.testing.assert_allclose(y_casadi, y_diffrax, rtol=check_tol, atol=check_tol)
        np.testing.assert_allclose(y_casadi, y_diffsol_bdf, rtol=check_tol, atol=check_tol)
        np.testing.assert_allclose(y_casadi, y_diffsol_esdirk34, rtol=check_tol, atol=check_tol)

        n = 500 // (int(0.01 * ng) + 1)
        print("ngroups: ", ng)
        print("tol: ", tol)
        print("n: ", n)
        casadi_time = timeit.timeit(casadi, number=n) / n
        print("Casadi time: ", casadi_time)
        diffsol_bdf_time = timeit.timeit(diffsol_bdf, number=n) / n
        print("Diffsol BDF time: ", diffsol_bdf_time)
        diffsol_esdirk34_time = timeit.timeit(diffsol_esdirk34, number=n) / n
        print("Diffsol ESDIRK34 time: ", diffsol_esdirk34_time)
        print("Speedup over casadi: ", casadi_time / diffsol_bdf_time)
        
        # Prepare result row
        result_row = {
            'ngroups': ng,
            'tolerance': tol,
            'n_runs': n,
            'casadi_time': casadi_time,
            'diffsol_bdf_time': diffsol_bdf_time,
            'diffsol_esdirk34_time': diffsol_esdirk34_time,
            'speedup_casadi_vs_bdf': casadi_time / diffsol_bdf_time,
            'diffrax_time': None,
            'speedup_diffrax_vs_bdf': None
        }
        
        if run_diffrax:
            diffrax_time = timeit.timeit(diffrax, number=n) / n
            print("Diffrax time: ", diffrax_time)
            print("Speedup over diffrax: ", diffrax_time / diffsol_bdf_time)
            result_row['diffrax_time'] = diffrax_time
            result_row['speedup_diffrax_vs_bdf'] = diffrax_time / diffsol_bdf_time
        
        # Add result to list
        results.append(result_row)

# Create DataFrame from results
df_results = pd.DataFrame(results)

# Display the results
print("\n" + "="*60)
print("BENCHMARK RESULTS SUMMARY")
print("="*60)
print(df_results.to_string(index=False))

# Save results to CSV
csv_filename = f"benchmark_results_robertson_ode.csv"
df_results.to_csv(csv_filename, index=False)
print(f"\nResults saved to: {csv_filename}")