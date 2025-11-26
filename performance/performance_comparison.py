import matplotlib.pyplot as plt
import numpy as np

# Set font to Arial
plt.rcParams["font.family"] = "Arial"

# Data from test_results.txt
test_runs = ["test 1 (13 images)", "test 2 (27 images)", "test 3 (250 images)"]
rt_detr_times = [115.63, 367.10, 2837.44]
detr_times = [75.57, 282.67, 2047.75]
yolo_times = [37.36, 199.11, 1331.20]

# Images per minute data
rt_detr_ipm = [6.63, 4.32, 5.00]
detr_ipm = [10.27, 5.67, 5.00]
yolo_ipm = [20.80, 8.10, 10.00]

# Total cropped objects data
rt_detr_objects = [87, 54, 1170]
detr_objects = [251, 60, 2192]
yolo_objects = [67, 52, 864]

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Set up bar positions
x = np.arange(len(test_runs))
width = 0.25

# Create grouped bars
bars1 = ax.bar(
    x - width,
    rt_detr_times,
    width,
    label="RT-DETR",
    color="#E63946",
    edgecolor="black",
    linewidth=1.2,
)
bars2 = ax.bar(
    x,
    detr_times,
    width,
    label="DETR",
    color="#A23B72",
    edgecolor="black",
    linewidth=1.2,
)
bars3 = ax.bar(
    x + width,
    yolo_times,
    width,
    label="YOLO",
    color="#2E86AB",
    edgecolor="black",
    linewidth=1.2,
)

# Customize the chart
ax.set_xlabel("Test Run", fontsize=12, fontweight="bold")
ax.set_ylabel("Total Time (seconds)", fontsize=12, fontweight="bold")
ax.set_title(
    "Object Detection Performance Comparison: RT-DETR vs DETR vs YOLO",
    fontsize=14,
    fontweight="bold",
    pad=20,
)
ax.set_xticks(x)
ax.set_xticklabels(test_runs)
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, alpha=0.3, linestyle="--", axis="y")

# Add value labels on bars with object count and images per minute
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{rt_detr_objects[i]} objects\n({rt_detr_ipm[i]} img/min)",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="normal",
    )

for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{detr_objects[i]} objects\n({detr_ipm[i]} img/min)",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="normal",
    )

for i, bar in enumerate(bars3):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{yolo_objects[i]} objects\n({yolo_ipm[i]} img/min)",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="normal",
    )

# Set y-axis to start from 0 for better comparison
ax.set_ylim(bottom=0, top=3000)

plt.tight_layout()
plt.savefig("performance_comparison.png", dpi=300, bbox_inches="tight")
print("Chart saved as 'performance_comparison.png'")
plt.show()
