import matplotlib.pyplot as plt
import numpy as np

# Set font to Arial
plt.rcParams["font.family"] = "Arial"

# Data from test_results.txt - 250_vehicles_open dataset only
test_runs = ["CPU (250 images)", "GPU (250 images)"]
rt_detr_times = [3566.40, 1306.11]

# Images per minute data
rt_detr_ipm = [4.21, 11.48]

# Total cropped objects data
rt_detr_objects = [1170, 1170]

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Set up bar positions
x = np.arange(len(test_runs))
width = 0.4

# Create bars
bars = ax.bar(
    x,
    rt_detr_times,
    width,
    label="RT-DETR",
    color=["#E63946", "#2E86AB"],
    edgecolor="black",
    linewidth=1.2,
)

# Customize the chart
ax.set_xlabel("Test Configuration", fontsize=12, fontweight="bold")
ax.set_ylabel("Total Time (seconds)", fontsize=12, fontweight="bold")
ax.set_title(
    "RT-DETR Performance: CPU vs GPU (250 images)",
    fontsize=14,
    fontweight="bold",
    pad=20,
)
ax.set_xticks(x)
ax.set_xticklabels(test_runs)
ax.grid(True, alpha=0.3, linestyle="--", axis="y")

# Add value labels on bars with object count and images per minute
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{rt_detr_times[i]:.2f}s\n{rt_detr_objects[i]} objects\n({rt_detr_ipm[i]} img/min)",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="normal",
    )

# Add speedup annotation
speedup = rt_detr_times[0] / rt_detr_times[1]
ax.text(
    0.5,
    max(rt_detr_times) * 0.9,
    f"GPU Speedup: {speedup:.2f}x faster",
    ha="center",
    fontsize=12,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
)

# Set y-axis to start from 0 for better comparison
ax.set_ylim(bottom=0, top=4000)

plt.tight_layout()
plt.savefig("performance_comparison_gpu.png", dpi=300, bbox_inches="tight")
print("Chart saved as 'performance_comparison_gpu.png'")
plt.show()
