import matplotlib

matplotlib.use("SVG")
import matplotlib.pyplot as plt
import pandas as pd


def plot():
    csv_filename = f"benchmark_results_robertson_ode.csv"
    csv_filename_jl = f"benchmark_results_robertson_ode_jl.csv"
    df = pd.read_csv(csv_filename)
    df_jl = pd.read_csv(csv_filename_jl)
    df = pd.merge(df, df_jl, on="ngroups", how="outer")
    computer_name = "Dell PowerEdge R7525 2U rack server"
    
    df_robertson_ode = df[df["

    fig, ax = plt.subplots(figsize=(10, 6))
    for method in [
        "casadi_time",
        "diffsol_bdf_time",
        "diffsol_esdirk34_time",
        "diffsol_tr_bdf2_time",
        "diffrax_time",
        "diffeq_bdf_time",
        "diffeq_kencarp3_time",
        "diffeq_tr_bdf2_time",
    ]:
        if method in df.columns:
            ax.plot(
                df["ngroups"] * 3,
                df[method],
                marker="o",
                label=method.replace("_time", "").replace("_", " ").title(),
            )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of states (log scale)")
    ax.set_ylabel("Time (seconds, log scale)")
    ax.set_title(f"Benchmark Results for stiff Robertson ODE on {computer_name}")
    ax.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("benchmark_robertson_ode.svg")


if __name__ == "__main__":
    plot()
