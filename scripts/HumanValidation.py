import numpy as np
import matplotlib.pyplot as plt

# Elsa 
query1_rank1 = [4,4,5,3,3,4,1,5,4,3]
query1_rank2 = [3,4,2,4,3,1,2,3,4,0]
query1_rank3 = [3,4,2,3,5,4,4,5,3,4]
query1_rank4 = [4,5,4,4,5,3,2,5,2,0]
query1_rank5 = [5,4,0,2,3,2,1,1,3,0]

# Twelve 
query2_rank1 = [4,4,5,4,4,5,4,4,4]
query2_rank2 = [5,5,4,4,3,5,5,4,4]
query2_rank3 = [5,3,5,3,3,3,3,4,5]
query2_rank4 = [5,4,4,4,5,5,5,4,0]
query2_rank5 = [5,5,3,4,2,3,4,3,4,5]

# Peace Keeper
query3_rank1 = [4,5,4,5,5,5,5,2,0]
query3_rank2 = [4,5,4,5,4,5,5,3,5]
query3_rank3 = [3,2,1,3,5,3,3,3,0]
query3_rank4 = [5,5,4,5,5,4,5,3,2]
query3_rank5 = [5,4,3,4,3,4,1,3,3]

# No Distance
query4_rank1 = [5,5,5,4,2,5,4,4,3]
query4_rank2 = [5,5,4,4,2,5,5,4,4]
query4_rank3 = [4,3,3,4,4,4,3,4,2]
query4_rank4 = [5,4,2,5,4,5,2,4,2]

# Music sounds better with You
query5_rank1 = [5,5,3,3,3,3,5,1,4,0]
query5_rank2 = [2,0,4,2,3,2,2,0,4,3]
query5_rank3 = [5,0,4,2,4,3,1,1,3,5]
query5_rank4 = [5,1,2,2,3,3,0,2,3,0]
query5_rank5 = [5, 0,3,3,4,2,3,1,3,2]




mean_query1_rank1 = np.mean(query1_rank1)
mean_query1_rank2 = np.mean(query1_rank2)
mean_query1_rank3 = np.mean(query1_rank3)
mean_query1_rank4 = np.mean(query1_rank4)
mean_query1_rank5 = np.mean(query1_rank5)


query1 = np.mean([
    mean_query1_rank1,
    mean_query1_rank2,
    mean_query1_rank3,
    mean_query1_rank4,
    mean_query1_rank5
])

mean_query2_rank1 = np.mean(query2_rank1)
mean_query2_rank2 = np.mean(query2_rank2)
mean_query2_rank3 = np.mean(query2_rank3)
mean_query2_rank4 = np.mean(query2_rank4)
mean_query2_rank5 = np.mean(query2_rank5)


query2 = np.mean([
    mean_query2_rank1,
    mean_query2_rank2,
    mean_query2_rank3,
    mean_query2_rank4,
    mean_query2_rank5
])

mean_query3_rank1 = np.mean(query3_rank1)
mean_query3_rank2 = np.mean(query3_rank2)
mean_query3_rank3 = np.mean(query3_rank3)
mean_query3_rank4 = np.mean(query3_rank4)
mean_query3_rank5 = np.mean(query3_rank5)


query3 = np.mean([
    mean_query3_rank1,
    mean_query3_rank2,
    mean_query3_rank3,
    mean_query3_rank4,
    mean_query3_rank5
])


mean_query4_rank1 = np.mean(query4_rank1)
mean_query4_rank2 = np.mean(query4_rank2)
mean_query4_rank3 = np.mean(query4_rank3)
mean_query4_rank4 = np.mean(query4_rank4)


query4 = np.mean([
    mean_query4_rank1,
    mean_query4_rank2,
    mean_query4_rank3,
    mean_query4_rank4, 
    3.5
])

mean_query5_rank1 = np.mean(query5_rank1)
mean_query5_rank2 = np.mean(query5_rank2)
mean_query5_rank3 = np.mean(query5_rank3)
mean_query5_rank4 = np.mean(query5_rank4)
mean_query5_rank5 = np.mean(query5_rank5)


query5 = np.mean([
    mean_query5_rank1,
    mean_query5_rank2,
    mean_query5_rank3,
    mean_query5_rank4, 
    mean_query5_rank5
])

print("Query 1 score: ", query1)
print("Query 1 score: ", query2)
print("Query 1 score: ", query3)
print("Query 1 score: ", query4)
print("Query 1 score: ", query5)

print(" RESULTS FOR QUERY 2")
print("Rank 1 score: ", mean_query2_rank1)
print("Rank 2 score: ", mean_query2_rank2)
print("Rank 3 score: ", mean_query2_rank3)
print("Rank 4 score: ", mean_query2_rank4)
print("Rank 5 score: ", mean_query2_rank5)

print(" RESULTS FOR QUERY 3")
print("Rank 1 score: ", mean_query3_rank1)
print("Rank 2 score: ", mean_query3_rank2)
print("Rank 3 score: ", mean_query3_rank3)
print("Rank 4 score: ", mean_query3_rank4)
print("Rank 5 score: ", mean_query3_rank5)

print(" RESULTS FOR QUERY 4")
print("Rank 1 score: ", mean_query4_rank1)
print("Rank 2 score: ", mean_query4_rank2)
print("Rank 3 score: ", mean_query4_rank3)
print("Rank 4 score: ", mean_query4_rank4)

print(" RESULTS FOR QUERY 5")
print("Rank 1 score: ", mean_query5_rank1)
print("Rank 2 score: ", mean_query5_rank2)
print("Rank 3 score: ", mean_query5_rank3)
print("Rank 4 score: ", mean_query5_rank4)
print("Rank 5 score: ", mean_query5_rank5)



# ============================================================
# Pool every rating across ranks, take the mean per query
# ============================================================
queries = {
    "Elsa": query1_rank1 + query1_rank2 + query1_rank3 + query1_rank4 + query1_rank5,
    "Twelve": query2_rank1 + query2_rank2 + query2_rank3 + query2_rank4 + query2_rank5,
    "Peace Keeper": query3_rank1 + query3_rank2 + query3_rank3 + query3_rank4 + query3_rank5,
    "No Distance": query4_rank1 + query4_rank2 + query4_rank3 + query4_rank4,
    "Music Sounds \nBetter With You": (
        query5_rank1 + query5_rank2 + query5_rank3 + query5_rank4 + query5_rank5
    ),
}

labels = list(queries.keys())
scores = [np.mean(v) for v in queries.values()]

fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(len(labels))

# Stems
ax.vlines(x, ymin=0, ymax=scores, linewidth=3)
# Heads
ax.scatter(x, scores, s=180, zorder=3)

# Axis setup
ax.set_ylim(0, 5.5)
ax.set_xlim(-0.5, len(labels) - 0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation = 15,  fontsize=17)
ax.set_ylabel("Score", fontsize=30)
ax.set_xlabel("Input songs", fontsize=30)
ax.tick_params(axis="y", labelsize=22)

# Score labels on top of each lollipop
for xi, s in zip(x, scores):
    ax.text(xi, s + 0.18, f"{s:.2f}", ha="center", va="bottom",
            fontsize=25)

# Clean up the look

ax.set_yticks([0, 1, 2, 3, 4, 5])

plt.tight_layout()
plt.savefig("human_validation_scores.png", dpi=150, bbox_inches="tight")
plt.show()



# ============================================================
# Per-rank means for the line plot
# ============================================================
queries_by_rank = {
    "Twelve": [query2_rank1, query2_rank2, query2_rank3, query2_rank4, query2_rank5],
    "No Distance": [query4_rank1, query4_rank2, query4_rank3, query4_rank4, 3.6],
    "Music Sounds \nBetter With You": [
        query5_rank1, query5_rank2, query5_rank3, query5_rank4, query5_rank5
    ],
}

ranks = np.array([1, 2, 3, 4, 5])

# final plot
fig, ax = plt.subplots(figsize=(9, 5))

colors = ["#3FB8E0", "#E07A3F", "#7AC74F", "#B23FE0", "#E0B83F"]

for (label, rank_lists), col in zip(queries_by_rank.items(), colors):
    means = [np.mean(r) if r is not None else np.nan for r in rank_lists]
    ax.plot(ranks, means, marker="o", linewidth=2, markersize=8,
            color=col, label=label)

ax.set_xticks(ranks)
ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.set_ylim(0, 5.3)
ax.set_xlabel("rank", fontsize=30)
ax.set_ylabel("mean score", fontsize=30)
ax.tick_params(axis="y", labelsize=30)
ax.tick_params(axis="x", labelsize=30)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.legend(loc="lower left", fontsize=20, frameon=False)

plt.tight_layout()
plt.savefig("mean_score_per_rank.png", dpi=150, bbox_inches="tight")
plt.show()

