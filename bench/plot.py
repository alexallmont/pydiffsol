import matplotlib

matplotlib.use("SVG")
import matplotlib.pyplot as plt
import pandas as pd


def plot():
    csv_filename = f"benchmark_results_robertson_ode.csv"
    csv_filename_jl = f"benchmark_results_robertson_ode_jl.csv"
    df = pd.read_csv(csv_filename)
    df_jl = pd.read_csv(csv_filename_jl)
    df = pd.merge(df, df_jl, on=["problem", "ngroups", "tol"], how="outer")
    # fill in the name of the local machine here
    computer_name = "Dell PowerEdge R7525 2U rack server"
    # computer_name = "Macbook Pro M2"
    
    df_robertson = df[df["problem"] == "robertson_ode"]
    df_lotka_volterra = df[df["problem"] == "lotka_volterra_ode"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for method in [
        "casadi_time",
        "diffsol_bdf_time",
        "diffsol_esdirk34_time",
        "diffsol_tr_bdf2_time",
        "diffrax_kvaerno5_time",
        "diffeq_bdf_time",
        "diffeq_kencarp3_time",
        "diffeq_tr_bdf2_time",
    ]:
        if method in df.columns:
            ax.plot(
                df_robertson["ngroups"] * 3,
                df_robertson[method],
                marker="o",
                label=method.replace("_time", "").replace("_", " ").title(),
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of states (log scale)")
    ax.set_ylabel("Time (seconds, log scale)")
    ax.set_title(f"Benchmark Results for stiff Robertson ODE on {computer_name}")
    ax.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("benchmark_robertson_ode.svg")

    fig, ax = plt.subplots(figsize=(10, 6))
    for method in [
        "casadi_time",
        "diffsol_bdf_time",
        "diffsol_tsit5_time",
        "diffrax_tsit5_time",
        "diffeq_bdf_time",
        "diffeq_tsit5_time",
    ]:
        if method in df.columns:
            ax.plot(
                df_lotka_volterra["tol"],
                df_lotka_volterra[method],
                marker="o",
                label=method.replace("_time", "").replace("_", " ").title(),
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tolerance (log scale)")
    ax.set_ylabel("Time (seconds, log scale)")
    ax.set_title(f"Benchmark Results for non-stiff Lotka-Volterra ODE on {computer_name}")
    ax.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("benchmark_lotka_volterra_ode.svg")



if __name__ == "__main__":
    plot()
