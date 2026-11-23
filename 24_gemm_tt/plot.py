import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

names = ["v2\ntiled16", "v3\n4x4", "v4\n+float4", "v5\n+occ67",
         "v6\n8x8", "v7\n+doublebuf", "v8\ndb+lb4"]
vals = [1593.7, 7729.2, 8085.4, 8692.9, 7854.8, 10311.7, 9636.5]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(names, vals, color=["#9aa5b1"] * len(vals))
bars[5].set_color("#1F4E79")

ax.axhline(20000, color="#c0392b", linestyle="--", linewidth=1)
ax.text(0.05, 20300, "peak ~20000 GFLOP/s", color="#c0392b", fontsize=9)

for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 250, "%.0f" % v,
            ha="center", fontsize=9)

ax.set_ylabel("GFLOP/s")
ax.set_title("SGEMM optimization path on RTX 4070 Laptop (n=2048)")
ax.set_ylim(0, 22500)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("gemm_curve.png", dpi=160)
print("saved gemm_curve.png")
