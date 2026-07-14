import networkx as nx
import numpy as np
import pandas as pd

from distfit import distfit
from scipy import stats
from statsmodels.stats.multitest import multipletests

from d3graph import d3graph, vec2adjmat

import matplotlib.pyplot as plt

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



def pagerank_permutation_test(
    adjmat,
    n_perm=1000,
    alpha=0.85,
    swaps_per_edge=10,
    random_state=42,
    fit_distributions=True,
    verbose=0,
):
    """Test PageRank scores using directed degree-preserving permutations.

    The null model preserves:
    - number of nodes
    - number of edges
    - in-degree of every node
    - out-degree of every node
    - global edge-weight distribution

    Parameters
    ----------
    adjmat : pandas.DataFrame
        Directed adjacency matrix. Rows are sources and columns are targets.

    n_perm : int, default=1000
        Number of randomized networks.

    alpha : float, default=0.85
        PageRank damping parameter.

    swaps_per_edge : int, default=10
        Requested edge swaps per edge for each randomized graph.

    random_state : int, default=42
        Random seed.

    fit_distributions : bool, default=True
        Fit a parametric null distribution with distfit for each node.

    verbose : int, default=0
        Print progress when larger than zero.

    Returns
    -------
    results : pandas.DataFrame
        Observed PageRank, empirical p-values, fitted p-values and FDR values.

    pagerank_null : pandas.DataFrame
        Null PageRank values. Rows are permutations and columns are nodes.

    fitted_models : dict
        Fitted distfit objects keyed by node.
    """
    if not isinstance(adjmat, pd.DataFrame):
        raise TypeError("adjmat must be a pandas DataFrame.")

    if adjmat.shape[0] != adjmat.shape[1]:
        raise ValueError("adjmat must be square.")

    if not adjmat.index.equals(adjmat.columns):
        raise ValueError(
            "The row and column labels of adjmat must have the same order."
        )

    rng = np.random.default_rng(random_state)

    # Build observed graph
    G = nx.from_pandas_adjacency(
        adjmat,
        create_using=nx.DiGraph,
    )

    # Remove zero-weight edges and self-loops
    zero_edges = [
        (source, target)
        for source, target, data in G.edges(data=True)
        if data.get("weight", 0) <= 0
    ]
    G.remove_edges_from(zero_edges)
    G.remove_edges_from(nx.selfloop_edges(G))

    nodes = list(adjmat.index)
    n_edges = G.number_of_edges()

    if n_edges < 3:
        raise ValueError(
            "At least three directed edges are required for directed edge swaps."
        )

    # Observed PageRank
    observed_dict = nx.pagerank(
        G,
        alpha=alpha,
        weight="weight",
    )

    observed = np.array(
        [observed_dict[node] for node in nodes],
        dtype=float,
    )

    # Preserve the global edge-weight distribution
    original_weights = np.array(
        [
            data.get("weight", 1.0)
            for _, _, data in G.edges(data=True)
        ],
        dtype=float,
    )

    pagerank_null = np.full(
        shape=(n_perm, len(nodes)),
        fill_value=np.nan,
        dtype=float,
    )

    nswap = max(1, swaps_per_edge * n_edges)
    max_tries = max(100, nswap * 20)

    for permutation in range(n_perm):
        G_null = G.copy()

        # Topology only: remove weights before rewiring
        for source, target in G_null.edges():
            G_null[source][target]["weight"] = 1.0

        try:
            nx.directed_edge_swap(
                G_null,
                nswap=nswap,
                max_tries=max_tries,
                seed=int(rng.integers(0, 2**32 - 1)),
            )
        except nx.NetworkXAlgorithmError:
            # Dense or highly constrained networks may not permit all swaps.
            # Retry with fewer requested swaps.
            nx.directed_edge_swap(
                G_null,
                nswap=max(1, nswap // 10),
                max_tries=max_tries,
                seed=int(rng.integers(0, 2**32 - 1)),
            )

        # Randomly assign original weights to the rewired edges
        shuffled_weights = rng.permutation(original_weights)

        for edge, weight in zip(G_null.edges(), shuffled_weights):
            source, target = edge
            G_null[source][target]["weight"] = float(weight)

        pr_null = nx.pagerank(
            G_null,
            alpha=alpha,
            weight="weight",
        )

        pagerank_null[permutation, :] = [
            pr_null[node] for node in nodes
        ]

        if verbose and (permutation + 1) % 100 == 0:
            print(
                f"Completed {permutation + 1}/{n_perm} permutations"
            )

    # Empirical upper-tail permutation p-value.
    #
    # Adding 1 prevents a zero p-value:
    # p = (number of null scores >= observed + 1) / (n_perm + 1)
    empirical_pvalue = (
        1
        + np.sum(
            pagerank_null >= observed[np.newaxis, :],
            axis=0,
        )
    ) / (n_perm + 1)

    null_mean = np.nanmean(pagerank_null, axis=0)
    null_std = np.nanstd(
        pagerank_null,
        axis=0,
        ddof=1,
    )

    null_zscore = np.divide(
        observed - null_mean,
        null_std,
        out=np.full_like(observed, np.nan),
        where=null_std > 0,
    )

    results = pd.DataFrame({
        "node": nodes,
        "pagerank": observed,
        "null_mean": null_mean,
        "null_std": null_std,
        "zscore": null_zscore,
        "pvalue_empirical": empirical_pvalue,
    })

    fitted_models = {}

    if fit_distributions:
        distribution_names = []
        fitted_pvalues = []

        for node_index, node in enumerate(nodes):
            null_values = pagerank_null[:, node_index]
            null_values = null_values[np.isfinite(null_values)]

            dfit = distfit(verbose=0)
            dfit.fit_transform(null_values)

            fitted_models[node] = dfit

            model_name = dfit.model["name"]
            model_params = dfit.model["params"]

            distribution = getattr(stats, model_name)

            # One-sided upper-tail probability:
            # probability of observing this PageRank or a larger one.
            pvalue = distribution.sf(
                observed[node_index],
                *model_params,
            )

            distribution_names.append(model_name)
            fitted_pvalues.append(float(pvalue))

        results["distribution"] = distribution_names
        results["pvalue_fitted"] = fitted_pvalues

        results["qvalue_fitted"] = multipletests(
            results["pvalue_fitted"],
            method="fdr_bh",
        )[1]

    results["qvalue_empirical"] = multipletests(
        results["pvalue_empirical"],
        method="fdr_bh",
    )[1]

    results["significant"] = (
        results["qvalue_empirical"] < 0.05
    )

    results = results.sort_values(
        ["qvalue_empirical", "pagerank"],
        ascending=[True, False],
    ).reset_index(drop=True)

    pagerank_null = pd.DataFrame(
        pagerank_null,
        columns=nodes,
    )

    return results, pagerank_null, fitted_models

# %%
d3 = d3graph()

df = d3.import_example('socialmedia')
adjmat = vec2adjmat(source=df['source'], target=df['target'], weight=df['weight'])
results, pagerank_null, fitted_models = pagerank_permutation_test(adjmat, n_perm=1000, swaps_per_edge=10, random_state=42, fit_distributions=True, verbose=1)

print(results[["node", "pagerank", "null_mean", "zscore", "pvalue_empirical", "qvalue_empirical", "distribution", "pvalue_fitted", "significant"]].head(20))

# The interpretation is:
# The observed PageRank of @important_user.social is higher than almost all PageRank scores obtained from randomized networks with exactly the same node-level in-degree and out-degree. Its position is therefore unlikely to be explained by degree alone.

# %%

node = results.iloc[0]["node"]
dfit = fitted_models[node]
dfit.plot(title=f"Null PageRank distribution: {node}")
observed_pagerank = results.loc[results["node"] == node, "pagerank"].iloc[0]

print("Observed PageRank:", observed_pagerank)
print("Empirical p-value:", results.loc[results["node"] == node, "pvalue_empirical"].iloc[0])

# %%

node_results = results.set_index("node").reindex(adjmat.index)

node_size = scale_values(
    node_results["pagerank"].values,
    min_value=8,
    max_value=45,
)

node_color = np.where(
    node_results["significant"].values,
    "#ff4d4d",
    "#808080",
).tolist()

node_tooltip = [
    (
        f"{node}<br>"
        f"PageRank: {row.pagerank:.5f}<br>"
        f"Expected: {row.null_mean:.5f}<br>"
        f"Z-score: {row.zscore:.2f}<br>"
        f"Empirical p: {row.pvalue_empirical:.4g}<br>"
        f"FDR q: {row.qvalue_empirical:.4g}<br>"
        f"Significant: {row.significant}"
    )
    for node, row in node_results.iterrows()
]


d3.graph(adjmat)

d3.set_node_properties(
    size=node_size,
    color=node_color,
    tooltip=node_tooltip,
)


# For reporting significance, I recommend using pvalue_empirical and qvalue_empirical as the primary results. The fitted distfit distribution is useful for smoothing and visualization, but the empirical permutation p-value makes fewer distributional assumptions.

