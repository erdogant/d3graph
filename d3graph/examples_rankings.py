from d3graph import d3graph, vec2adjmat

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def scale_values(values, min_value=8, max_value=40):
    """Scale numeric values to an integer range."""
    values = np.asarray(values, dtype=float)

    if np.allclose(values.min(), values.max()):
        return np.full(values.shape, min_value, dtype=int)

    scaled = (
        min_value
        + (values - values.min())
        * (max_value - min_value)
        / (values.max() - values.min())
    )

    return np.round(scaled).astype(int)


def values_to_colors(values, cmap="viridis"):
    """Convert numeric values to hexadecimal colors."""
    values = np.asarray(values, dtype=float)

    if np.allclose(values.min(), values.max()):
        normalized = np.zeros_like(values)
    else:
        normalized = (
            (values - values.min())
            / (values.max() - values.min())
        )

    colormap = plt.get_cmap(cmap)

    return [
        "#{:02x}{:02x}{:02x}".format(
            int(r * 255),
            int(g * 255),
            int(b * 255),
        )
        for r, g, b, _ in colormap(normalized)
    ]


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
d3 = d3graph()
df = d3.import_example("socialmedia")

# Slice the first 1,000 interactions
df = df.iloc[:10000].copy()
# Create adjacency matrix
adjmat = vec2adjmat(source=df["source"], target=df["target"], weight=df["weight"])


# ---------------------------------------------------------
# Compute weighted HITS
# ---------------------------------------------------------
G = nx.from_pandas_adjacency(adjmat, create_using=nx.DiGraph)
hub_scores, authority_scores = nx.hits(G, max_iter=1000, tol=1e-8, normalized=True)

# Make sure scores follow exactly the adjacency-matrix order
nodes = adjmat.index.tolist()

scores = pd.DataFrame({
    "node": nodes,
    "hub_score": [hub_scores[node] for node in nodes],
    "authority_score": [
        authority_scores[node]
        for node in nodes
    ],
})

scores["combined_score"] = (scores["hub_score"] + scores["authority_score"]) / 2


# ---------------------------------------------------------
# Convert scores into visual properties
# ---------------------------------------------------------

# Large node = strong authority
node_size = scale_values(scores["authority_score"], min_value=8, max_value=100)

# Bright color = strong hub
node_color = values_to_colors(
    scores["hub_score"],
    cmap="plasma",
)

node_tooltip = [
    (
        f"{row.node}<br>"
        f"Authority: {row.authority_score:.4f}<br>"
        f"Hub: {row.hub_score:.4f}<br>"
        f"Combined: {row.combined_score:.4f}"
    )
    for row in scores.itertuples()
]


# ---------------------------------------------------------
# Create d3graph
# ---------------------------------------------------------
d3.graph(adjmat)
d3.set_node_properties(size=node_size, color=node_color, tooltip=node_tooltip)

# ---------------------------------------------------------
# Show graph
# ---------------------------------------------------------
d3.show(
    density_grid_size=150,
    density_blur=10,
    density_opacity=0.6,
    dark_mode=True,
    show_density=True,
    show_slider=True,
    show_controls=True,
)

# %%

top_authorities = scores.sort_values("authority_score", ascending=False).head(10)
top_hubs = scores.sort_values("hub_score", ascending=False).head(10)

print("Top authorities")
print(
    top_authorities[
        ["node", "authority_score", "hub_score"]
    ]
)

print("\nTop hubs")
print(
    top_hubs[
        ["node", "hub_score", "authority_score"]
    ]
)

# These nodes have the highest hub scores, which means they are good connectors rather than necessarily being the most influential or popular accounts.

# In HITS:
# Hub score answers: "Does this account point to important accounts?"
# Authority score answers: "Do important accounts point to this account?"

# | Node                    |          Hub | Authority | Interpretation                                                                       |
# | ----------------------- | -----------: | --------: | ------------------------------------------------------------------------------------ |
# | @Uzbekmastodon.social   | **0.014656** |  0.001477 | Connects to many important accounts but is not itself frequently referenced by them. |
# | @you_laugh2@...         | **0.013841** |  0.000501 | Acts as a curator or broadcaster, directing attention toward authoritative users.    |
# | @alegrilmastodon.social | **0.013661** |  0.000205 | Similar role: a strong connector but not a central authority.                        |


# Intuition
# Imagine a conference:
# Authorities are the keynote speakers everyone refers to.
# Hubs are the attendees who know all the keynote speakers and introduce people to them.

# A hub doesn't have to be famous—it becomes valuable because it connects others to influential people.
# Why are the authority scores so small?
# This is perfectly normal. HITS computes the dominant eigenvectors of the graph, and the absolute values themselves have no intrinsic meaning. What matters is the ranking.

# A good way to explain this in your blog

# HITS distinguishes two different notions of importance. Authorities are users that receive attention from well-connected users, 
# while hubs are users that actively connect to many authorities.
# In social networks, authorities often represent influential individuals, whereas hubs act more like curators, aggregators, 
# or information distributors.

# This distinction is one of the main advantages of HITS over methods like PageRank, which produces a single influence score.
# HITS reveals different roles that users play within the network rather than collapsing everything into one ranking.



# %%
# =============================================================================
# PAGERANKS
# =============================================================================
import networkx as nx
import pandas as pd

# Create directed graph from adjacency matrix
G = nx.from_pandas_adjacency(adjmat, create_using=nx.DiGraph)

# Compute weighted PageRank
pagerank = nx.pagerank(G, alpha=0.85, weight="weight")

# Convert to dataframe
scores = (
    pd.DataFrame({
        "node": list(pagerank.keys()),
        "pagerank": list(pagerank.values())
    })
    .sort_values("pagerank", ascending=False)
    .reset_index(drop=True)
)

print(scores.head(10))


# Node size
node_size = scale_values(
    scores["pagerank"],
    min_value=8,
    max_value=45,
)

# Node color
node_color = values_to_colors(
    scores["pagerank"],
    cmap="plasma",
)

node_tooltip = [
    f"{row.node}<br>PageRank: {row.pagerank:.5f}"
    for row in scores.itertuples()
]

d3.graph(adjmat)

d3.set_node_properties(
    size=node_size,
    color=node_color,
    tooltip=node_tooltip,
)

d3.show(
    density_grid_size=150,
    density_blur=10,
    density_opacity=0.6,
    dark_mode=True,
    show_density=True,
    show_slider=True,
    show_controls=True,
)

# %%
# For a social network, permute the edges while preserving the number of nodes, or even better, preserve the degree distribution if you want a stronger null model. For a blog, simple edge permutation is easy to explain.

# 1 Compute the observed PageRank.
# 2 Destroy the network structure by random permutation.
# 3 Compute PageRank again.
# 4 Repeat many times (e.g., 1000).
# 5 This gives a null distribution for every node.
# 6 Use distfit to estimate the null distribution (or use the empirical distribution directly).
# 7 Compute p-values and FDR.

G = nx.from_pandas_adjacency(adjmat, create_using=nx.DiGraph)
pagerank_obs = nx.pagerank(G, alpha=0.85, weight="weight")
nodes = list(adjmat.index)
pagerank_obs = np.array([pagerank_obs[n] for n in nodes])

# %%
# =============================================================================
# PERMUTE
# =============================================================================

n_perm = 100
pagerank_null = np.zeros((n_perm, len(nodes)))
adj = adjmat.values.copy()

for i in range(n_perm):
    # Randomize edge locations
    shuffled = adj.flatten().copy()
    np.random.shuffle(shuffled)
    shuffled = shuffled.reshape(adj.shape)
    G_perm = nx.from_numpy_array(shuffled, create_using=nx.DiGraph)
    pr = nx.pagerank(G_perm, alpha=0.85, weight="weight")
    pagerank_null[i] = np.array([pr[k] for k in range(len(nodes))])

# %%
# =============================================================================
#  Use distfit to estimate the null distribution (or use the empirical distribution directly).
# =============================================================================
from distfit import distfit

results = []

for i, node in enumerate(nodes):
    dfit = distfit(verbose=0)
    dfit.fit_transform(pagerank_null[:, i])
    p = dfit.predict(pagerank_obs[i])["y_proba"]
    results.append([node, pagerank_obs[i], p, dfit.model["name"]])

df = pd.DataFrame(results, columns=["node", "pagerank", "pvalue", "distribution"])

# %%
node = 0

dfit = distfit()
dfit.fit_transform(pagerank_null[:, node])

dfit.plot()
dfit.plot_summary()
node = 0

dfit = distfit()
dfit.fit_transform(pagerank_null[:, node])

dfit.plot()
dfit.plot_summary()