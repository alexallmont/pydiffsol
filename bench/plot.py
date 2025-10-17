import matplotlib
matplotlib.use('SVG')
import matplotlib.pyplot as plt
import pandas as pd



csv_filename = f"benchmark_results_robertson_ode.csv"
df = pd.read_csv(csv_filename)
computer_name = "Dell PowerEdge R7525 2U rack server"

fig, ax = plt.subplots(figsize=(10, 6))
for method in ['casadi_time', 'diffsol_bdf_time', 'diffsol_esdirk34_time', 'diffrax_time']:
    if method in df.columns:
        ax.plot(df['ngroups'] * 3, df[method], marker='o', label=method.replace('_time', '').replace('_', ' ').title())
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xlabel('Number of states (log scale)')
ax.set_ylabel('Time (seconds, log scale)')
ax.set_title(f'Benchmark Results for stiff Robertson ODE on {computer_name}')
ax.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.savefig('benchmark_robertson_ode.svg')