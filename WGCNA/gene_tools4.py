import numpy as np
import pandas as pd
import threading
from fastmcp import FastMCP
from typing import Annotated, Literal
from pydantic import Field
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
import requests
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

mcp = FastMCP(name="Gene-CoExpression-WGCNA")

# --------------------------------------------------
# GLOBAL STATE
# --------------------------------------------------

EXPR_DATA = pd.DataFrame()
EXPR_DATA_INITIAL = pd.DataFrame()
TRAIT_DATA = pd.DataFrame()
RESULTS = {}

bloque2 = '''
load_data()

'''

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

@mcp.tool()
async def load_data() -> str:
    """
    Loads gene expression data and sample traits from local CSV files
    into the global server state.

    Expected format:
    - Expression matrix: rows = samples, columns = genes
    - Trait matrix: rows = samples, columns = traits

    Only samples present in both datasets are retained.
    Only numeric traits are kept for downstream correlation analysis.
    """
    global EXPR_DATA_INITIAL, TRAIT_DATA

    df_expr = pd.read_csv("expresion_wgcna_multi_modulo2.csv", index_col=0)
    df_traits = pd.read_csv("metadata_pacientes2.csv", index_col=0)

    common_samples = df_expr.index.intersection(df_traits.index)

    EXPR_DATA_INITIAL = df_expr.loc[common_samples].copy()
    TRAIT_DATA = df_traits.loc[common_samples].select_dtypes(include=[np.number]).copy()

    return (
        f"Data loaded successfully.\n"
        f"- Samples: {EXPR_DATA_INITIAL.shape[0]}\n"
        f"- Genes: {EXPR_DATA_INITIAL.shape[1]}\n"
        f"- Numeric traits: {list(TRAIT_DATA.columns)}"
    )


# --- DATASET STATISTICS ---
@mcp.tool()
async def get_data_statistics() -> str:
    """
    Computes a detailed variance profile of genes to help decide
    an appropriate min_var_quantile for WGCNA preprocessing.
    
    Returns summary statistics and gene counts at common quantile cutoffs.
    """

    global EXPR_DATA_INITIAL

    if EXPR_DATA_INITIAL.empty:
        return "Expression data not loaded."

    # Compute gene variance (across samples)
    gene_var = EXPR_DATA_INITIAL.var(axis=0)

    # Basic summary statistics
    summary = {
        "min": float(gene_var.min()),
        "q5": float(gene_var.quantile(0.05)),
        "q10": float(gene_var.quantile(0.10)),
        "q25": float(gene_var.quantile(0.25)),
        "median": float(gene_var.median()),
        "q75": float(gene_var.quantile(0.75)),
        "q90": float(gene_var.quantile(0.90)),
        "q95": float(gene_var.quantile(0.95)),
        "max": float(gene_var.max()),
        "mean": float(gene_var.mean()),
        "std": float(gene_var.std())
    }

    # Fraction of near-zero variance genes
    near_zero_threshold = 1e-5
    low_variance_fraction = float((gene_var < near_zero_threshold).mean())

    # Genes retained at typical WGCNA quantile cutoffs
    quantile_options = [0.1, 0.25, 0.5, 0.75]
    retained_counts = {}

    for q in quantile_options:
        cutoff = gene_var.quantile(q)
        retained_counts[f"retain_top_{int((1-q)*100)}pct"] = int((gene_var > cutoff).sum())

    result = {
        "total_genes": int(len(gene_var)),
        "variance_summary": summary,
        "low_variance_fraction": low_variance_fraction,
        "retained_gene_counts_by_quantile": retained_counts,
        "recommended_min_genes_for_WGCNA": 3000
    }

    return str(result)



# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

@mcp.tool(tags=["PREP"])
async def filter_genes_by_variance(
    min_var_quantile: Annotated[
        float,
        Field(
            description=(
                "Quantile threshold (0–1) for gene variance filtering. "
                "Genes with variance below this quantile are removed to "
                "discard low-information genes."
            ),
            ge=0.0,
            le=1.0,
        ),
    ]
) -> str:
    """
    Filters genes based on variance across samples.

    This step removes genes with low variability, which typically
    contribute little to co-expression structure and can negatively
    affect network topology estimation.
    """
    global EXPR_DATA, EXPR_DATA_INITIAL, bloque2

    variances = EXPR_DATA_INITIAL.var(axis=0)
    threshold = variances.quantile(min_var_quantile)
    keep = variances[variances >= threshold].index

    EXPR_DATA = EXPR_DATA_INITIAL[keep]

    bloque2 += f'''
aux = filter_genes_by_variance({min_var_quantile})
print(aux)
    '''

    return (
        f"Variance filtering applied.\n"
        f"- Quantile used: {min_var_quantile}\n"
        f"- Genes retained: {len(keep)}"
    )


@mcp.tool(tags=["PREP"])
async def sample_outlier_scores() -> str:
    """
    Computes a simple sample-level outlier score based on
    average distance from the sample correlation centroid.

    Larger values indicate samples that are globally less
    correlated with the rest of the dataset.
    """
    global EXPR_DATA, bloque2

    sample_cor = EXPR_DATA.T.corr()
    dist = (1 - sample_cor).mean(axis=1)
    bloque2 += f'''
aux = sample_outlier_scores()
print("Outliers:"+aux)
'''
    return dist.sort_values(ascending=False).to_string()

# --------------------------------------------------
# SIMILARITY MEASUREMENT
# --------------------------------------------------

@mcp.tool(tags=["SIMIL"])
async def compute_correlation(
    method: Annotated[
        Literal["pearson", "spearman"],
        Field(
            description=(
                "Correlation method used to measure gene–gene similarity. "
            )
        ),
    ] = "pearson"
) -> str:
    """
    Computes the gene–gene correlation matrix from the
    filtered expression data.
    """
    global EXPR_DATA, RESULTS, bloque2

    bloque2 += f'''
aux = compute_correlation({method})
print(aux)
    '''
    RESULTS["cor_matrix"] = EXPR_DATA.corr(method=method)

    return f"{method.capitalize()} correlation matrix computed."

# --------------------------------------------------
# SOFT-THRESHOLD SELECTION
# --------------------------------------------------

def scale_free_fit_index(adj: np.ndarray,
                         n_bins: int = 20,
                         min_points: int = 3):
    """
    Compute the scale-free topology fit index (R^2) 
    following the logic of the original WGCNA implementation.

    Parameters
    ----------
    adj : np.ndarray
        Symmetric adjacency matrix.
    n_bins : int
        Number of bins in log10(k) space.
    min_points : int
        Minimum number of valid bins required for regression.

    Returns
    -------
    r2 : float
        Scale-free topology fit index (R^2).
    mean_k : float
        Mean network connectivity.
    """

    # ---- 1. Safety checks ----
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Adjacency matrix must be square.")

    # ---- 3. Compute connectivity ----
    k = np.sum(adj, axis=1) - 1

    k = np.maximum(k, 1e-10)

    mean_k = np.mean(k)

    # ---- 4. Log-transform connectivity ----
    log_k = np.log10(k)

    # ---- 5. Histogram in log space ----
    counts, bin_edges = np.histogram(log_k, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Keep only non-empty bins
    mask = counts > 0
    if np.sum(mask) < min_points:
        return np.nan, mean_k

    x = bin_centers[mask]
    y = np.log10(counts[mask] / np.sum(counts))

    # ---- 6. Linear regression in log–log space ----
    #slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    #r2 = r_value ** 2

    r_matrix = np.corrcoef(x, y)
    r2 = r_matrix[0, 1]**2

    return r2, mean_k



@mcp.tool(tags=["NETWORK"])
async def pick_soft_threshold_power(
    network_type: Annotated[
        Literal["unsigned", "signed"],
        Field(
            description=(
                "Type of co-expression network. "
                "'unsigned' uses |correlation|^beta, "
                "'signed' uses ((1 + correlation) / 2)^beta."
            )
        ),
    ] = "unsigned"
) -> str:
    """
    Evaluates candidate soft-thresholding powers by computing
    scale-free topology fit (R^2) and mean connectivity.

    This tool does not select a final beta automatically; it
    exposes the full diagnostic table so that the decision can
    be made externally.
    """
    global RESULTS, bloque2

    cor = RESULTS.get("cor_matrix")
    powers = [int(p) for p in range(1,20)]

    rows = []
    for beta in powers:
        if network_type == "unsigned":
            adj = np.abs(cor.values) ** beta
        else:
            adj = ((1 + cor.values) / 2) ** beta

        r2, mean_k = scale_free_fit_index(adj)
        rows.append({
            "beta": beta,
            "scale_free_R2": r2,
            "mean_connectivity": mean_k
        })

    df = pd.DataFrame(rows)
    RESULTS["beta_table"] = df
    
    # Filtro de seguridad para evitar que el modelo elija ruido
    #filtered_df = df[df["scale_free_R2"] >= 0.75]
    filtered_df = df
    
    bloque2 += f'''
aux = pick_soft_threshold_power({network_type})
print("Betas:"+aux)
    '''

    if filtered_df.empty:
        top_val = df.sort_values("scale_free_R2", ascending=False).head(3)
        return f"CRITICAL WARNING: No power reached R^2 >= 0.75. Data might be noisy. Best candidates:\n{top_val.to_string(index=False)}"
    
    return filtered_df.to_string(index=False)



# --------------------------------------------------
# NETWORK CONSTRUCTION & MODULE DETECTION
# --------------------------------------------------

@mcp.tool(tags=["CLUSTERING"])
async def build_network_and_detect_modules(
    beta: Annotated[
        int,
        Field(description="Soft-thresholding power selected from the previous step.", gt=0)
    ],
    network_type: Annotated[
        Literal["unsigned", "signed"],
        Field(description="Network type used to compute adjacency.")
    ] = "unsigned",
    cut_height: Annotated[
        float,
        Field(
            description=(
                "Tree cut height applied to TOM-based hierarchical clustering. "
                "Typical values are between 0.90 and 0.99."
            ),
            gt=0.0,
            lt=1.0,
        ),
    ] = 0.95,
    min_module_size: Annotated[
        int,
        Field(
            description=(
                "Minimum number of genes per module. "
                "Smaller clusters are reassigned to the grey (0) module."
            ),
            gt=5,
        ),
    ] = 30
) -> str:
    """
    Builds the adjacency and TOM matrices and performs hierarchical
    clustering to identify co-expression modules.
    """
    global RESULTS, bloque2

    cor = RESULTS["cor_matrix"]

    if network_type == "unsigned":
        adj = np.abs(cor.values) ** beta
    else:
        adj = ((1 + cor.values) / 2) ** beta

    np.fill_diagonal(adj, 0.0)

    k = adj.sum(axis=0)
    L = adj @ adj
    denom = np.minimum.outer(k, k) + 1 - adj
    tom = (L + adj) / np.maximum(denom, 1e-12)
    np.fill_diagonal(tom, 1.0)

    diss = 1 - tom
    Z = linkage(squareform(diss, checks=False), method="average")
    # Aplicamos el clustering
    clusters = fcluster(Z, t=cut_height, criterion="distance")
    labels = pd.Series(clusters, index=cor.index)
    
    # Consolidación: Tamaño mínimo estricto
    counts = labels.value_counts()
    small = counts[counts < min_module_size].index
    labels = labels.where(~labels.isin(small), other=0)
    
    RESULTS["labels"] = labels
    RESULTS["Z"] = Z
    RESULTS["beta"] = beta
    RESULTS["network_type"] = network_type
    RESULTS["cut_height"] = cut_height
    RESULTS["min_module_size"] = min_module_size

    bloque2 += f'''
aux = build_network_and_detect_modules({beta},{network_type},{cut_height},{min_module_size})
print(aux)
    '''

        
    return f"Module detection completed. Found {len(labels.unique())-1} modules. Grey genes: {counts.get(0, 0)}"


# --------------------------------------------------
# MODULE–TRAIT ASSOCIATION
# --------------------------------------------------

@mcp.tool(tags=["ASSOCIATION"])
async def correlate_modules_with_traits() -> str:
    """
    Computes module eigengenes (first principal component of each module)
    and correlates them with numeric sample traits using Pearson correlation.

    The grey module (0) is excluded from this analysis.
    """
    global RESULTS, TRAIT_DATA, EXPR_DATA, bloque2

    labels = RESULTS["labels"]
    traits = TRAIT_DATA

    results = []
    for mod in sorted(labels.unique()):
        if mod == 0:
            continue

        genes = labels[labels == mod].index
        X = EXPR_DATA[genes].values
        X -= X.mean(axis=0, keepdims=True)

        me = PCA(n_components=1).fit_transform(X).ravel()

        for tr in traits.columns:
            r, p = stats.pearsonr(me, traits[tr])
            results.append({
                "module": mod,
                "trait": tr,
                "r": r,
                "p": p
            })

    df = pd.DataFrame(results)
    RESULTS["module_trait_correlations"] = df

    bloque2 += f'''
aux = correlate_modules_with_traits()
print("Correlation to traits:"+aux)
    '''

    return df.sort_values("p").to_string(index=False)

# --------------------------------------------------
# ORCHESTRATION TOOLS
# --------------------------------------------------

@mcp.tool(tags=["MOVE"])
async def move_to_next_step(summary: str) -> str:
    """Call this when you are satisfied with the results of the current step."""
    return f"TRANSITION_APPROVED: {summary}"


def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        return img


# @mcp.tool(tags=["DONE"])
# async def expose_results_to_web() -> str:
#     """
#     Exports data required to reproduce the final visualization:
#     - Dendrogram
#     - Module count
#     - Module–trait heatmap
#     """

#     required = ["labels", "Z"]
#     for key in required:
#         if key not in RESULTS:
#             return f"Error: Missing '{key}' in RESULTS. Run module detection first."

#     labels = RESULTS["labels"]
#     Z = RESULTS["Z"]
#     df_corr = RESULTS.get("module_trait_correlations", None)

#     # ---- Module summary ----
#     module_sizes = labels.value_counts()
#     n_modules = len(module_sizes) - (1 if 0 in module_sizes.index else 0)
#     grey_genes = int(module_sizes.get(0, 0))

#     module_size_list = [
#         {"module": int(mod), "size": int(size)}
#         for mod, size in module_sizes.items()
#     ]

#     # ---- Dendrogram data ----
#     dendrogram_payload = {
#         "linkage_matrix": Z.tolist()
#     }

#     # ---- Heatmap data (if correlations exist) ----
#     heatmap_payload = None
#     top_module = None

#     if df_corr is not None and not df_corr.empty:
#         # Pivot para matrices r y p
#         pivot_r = df_corr.pivot(index="module", columns="trait", values="r")
#         pivot_p = df_corr.pivot(index="module", columns="trait", values="p")

#         heatmap_payload = {
#             "r_matrix": pivot_r.values.tolist(),
#             "p_matrix": pivot_p.values.tolist(),
#             "modules": pivot_r.index.astype(int).tolist(),
#             "traits": pivot_r.columns.tolist()
#         }

#         # Mejor módulo: correlación absoluta más alta
#         top_row = df_corr.loc[df_corr["r"].abs().idxmax()]
#         top_module = {
#             "module": int(top_row["module"]),
#             "trait": str(top_row["trait"]),
#             "r": float(top_row["r"]),
#             "p": float(top_row["p"])
#         }

#     payload = {
#         "summary": {
#             "n_modules": int(n_modules),
#             "grey_genes": grey_genes,
#             "total_genes": int(len(labels)),
#             "beta": RESULTS.get("beta"),
#             "cut_height": RESULTS.get("cut_height"),
#         },
#         "module_sizes": module_size_list,
#         "dendrogram": dendrogram_payload,
#         "heatmap": heatmap_payload,
#         "top_module": top_module
#     }

#     target_url = "http://localhost:5001/update_results"

#     try:
#         response = requests.post(target_url, json=payload, timeout=10)

#         if response.status_code == 200:
#             return (
#                 f"✅ SUCCESS: Full analysis exported.\n"
#                 f"- Modules detected: {n_modules}\n"
#                 f"- Grey genes: {grey_genes}\n"
#                 f"- Genes: {len(labels)}\n"
#                 f"- Beta used: {RESULTS.get('beta')}\n"
#                 f"- Dashboard running on port 5001."
#             )
#         else:
#             return f"❌ ERROR: Dashboard responded with {response.status_code}."

#     except requests.exceptions.ConnectionError:
#         return (
#             "❌ ERROR: Could not connect to dashboard.\n"
#             "Make sure visualizer_app.py is running on port 5001."
#         )
#     except Exception as e:
#         return f"❌ ERROR: Unexpected error: {str(e)}"

@mcp.tool(tags=["DONE"])
async def expose_results_to_web() -> str:
    """
    Exports data required to reproduce the final visualization:
    - Dendrogram (data + image base64)
    - Module count
    - Module–trait heatmap (data + image base64)
    """

    required = ["labels", "Z"]
    for key in required:
        if key not in RESULTS:
            return f"Error: Missing '{key}' in RESULTS. Run module detection first."

    labels = RESULTS["labels"]
    Z = RESULTS["Z"]
    df_corr = RESULTS.get("module_trait_correlations", None)

    # ---- Module summary ----
    module_sizes = labels.value_counts()
    n_modules = len(module_sizes) - (1 if 0 in module_sizes.index else 0)
    grey_genes = int(module_sizes.get(0, 0))

    module_size_list = [
        {"module": int(mod), "size": int(size)}
        for mod, size in module_sizes.items()
    ]

    # =================================================
    # 🌳 DENDROGRAM DATA + IMAGE
    # =================================================
    dendrogram_payload = {
        "linkage_matrix": Z.tolist()
    }

    dendrogram_img = None
    try:
        fig = plt.figure(figsize=(12, 6))
        dendrogram(
            Z,
            no_labels=True,
            color_threshold=RESULTS.get("cut_height", 0.95)
        )
        plt.title("Gene Dendrogram")
        plt.xlabel("Genes")
        plt.ylabel("Distance")

        dendrogram_img = fig_to_base64(fig)
        plt.close(fig)
    except Exception:
        dendrogram_img = None

    # =================================================
    # 🔥 HEATMAP DATA + IMAGE
    # =================================================
    heatmap_payload = None
    heatmap_img = None
    top_module = None

    if df_corr is not None and not df_corr.empty:

        pivot_r = df_corr.pivot(index="module", columns="trait", values="r")
        pivot_p = df_corr.pivot(index="module", columns="trait", values="p")

        heatmap_payload = {
            "r_matrix": pivot_r.values.tolist(),
            "p_matrix": pivot_p.values.tolist(),
            "modules": pivot_r.index.astype(int).tolist(),
            "traits": pivot_r.columns.tolist()
        }

        # ---- Heatmap Image ----
        try:
            annot = pivot_r.round(2).astype(str) + "\n(p=" + pivot_p.round(3).astype(str) + ")"

            fig = plt.figure(figsize=(10, 6))
            sns.heatmap(
                pivot_r,
                annot=annot,
                fmt="",
                center=0
            )
            plt.title("Module-Trait Relationships")

            heatmap_img = fig_to_base64(fig)
            plt.close(fig)
        except Exception:
            heatmap_img = None

        # ---- Top module ----
        top_row = df_corr.loc[df_corr["r"].abs().idxmax()]
        top_module = {
            "module": int(top_row["module"]),
            "trait": str(top_row["trait"]),
            "r": float(top_row["r"]),
            "p": float(top_row["p"])
        }

    # =================================================
    # 📦 FINAL PAYLOAD
    # =================================================
    payload = {
        "summary": {
            "n_modules": int(n_modules),
            "grey_genes": grey_genes,
            "total_genes": int(len(labels)),
            "beta": RESULTS.get("beta"),
            "cut_height": RESULTS.get("cut_height"),
        },
        "module_sizes": module_size_list,
        "dendrogram": dendrogram_payload,
        "heatmap": heatmap_payload,
        "top_module": top_module,

        # 🔥 NUEVO
        "plots": {
            "dendrogram": dendrogram_img,
            "heatmap": heatmap_img
        }
    }

    target_url = "http://localhost:5001/update_results"

    try:
        response = requests.post(target_url, json=payload, timeout=10)

        if response.status_code == 200:
            return (
                f"✅ SUCCESS: Full analysis exported.\n"
                f"- Modules detected: {n_modules}\n"
                f"- Grey genes: {grey_genes}\n"
                f"- Genes: {len(labels)}\n"
                f"- Beta used: {RESULTS.get('beta')}\n"
                f"- Dashboard running on port 5001."
            )
        else:
            return f"❌ ERROR: Dashboard responded with {response.status_code}."

    except requests.exceptions.ConnectionError:
        return (
            "❌ ERROR: Could not connect to dashboard.\n"
            "Make sure visualizer_app.py is running on port 5001."
        )
    except Exception as e:
        return f"❌ ERROR: Unexpected error: {str(e)}"


@mcp.tool(tags=["DONE"])
async def expose_llm_summary_to_web(summary_text: str) -> str:

    target_url = "http://localhost:5001/update_summary"

    try:
        response = requests.post(target_url, json={
            "summary_text": summary_text
        }, timeout=10)

        if response.status_code == 200:
            return "✅ LLM summary sent to dashboard"
        else:
            return f"❌ Dashboard error: {response.status_code}"

    except Exception as e:
        return f"❌ Connection error: {str(e)}"

@mcp.tool(tags=["DONE"])
async def notebook_creation() -> str:
    bloque1 = '''
    import numpy as np
    import pandas as pd
    import threading
    from typing import Annotated, Literal
    from pydantic import Field
    from scipy import stats
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from sklearn.decomposition import PCA



    EXPR_DATA = pd.DataFrame()
    EXPR_DATA_INITIAL = pd.DataFrame()
    TRAIT_DATA = pd.DataFrame()
    RESULTS = {}

    async def load_data() -> str:
        """
        Loads gene expression data and sample traits from local CSV files
        into the global server state.

        Expected format:
        - Expression matrix: rows = samples, columns = genes
        - Trait matrix: rows = samples, columns = traits

        Only samples present in both datasets are retained.
        Only numeric traits are kept for downstream correlation analysis.
        """
        global EXPR_DATA_INITIAL, TRAIT_DATA

        df_expr = pd.read_csv("expresion_wgcna_multi_modulo2.csv", index_col=0)
        df_traits = pd.read_csv("metadata_pacientes2.csv", index_col=0)

        common_samples = df_expr.index.intersection(df_traits.index)

        EXPR_DATA_INITIAL = df_expr.loc[common_samples].copy()
        TRAIT_DATA = df_traits.loc[common_samples].select_dtypes(include=[np.number]).copy()

        return (
            f"Data loaded successfully.\n"
            f"- Samples: {EXPR_DATA_INITIAL.shape[0]}\n"
            f"- Genes: {EXPR_DATA_INITIAL.shape[1]}\n"
            f"- Numeric traits: {list(TRAIT_DATA.columns)}"
        )



    async def filter_genes_by_variance(
        min_var_quantile: Annotated[
            float,
            Field(
                description=(
                    "Quantile threshold (0–1) for gene variance filtering. "
                    "Genes with variance below this quantile are removed to "
                    "discard low-information genes."
                ),
                ge=0.0,
                le=1.0,
            ),
        ]
    ) -> str:
        """
        Filters genes based on variance across samples.

        This step removes genes with low variability, which typically
        contribute little to co-expression structure and can negatively
        affect network topology estimation.
        """
        global EXPR_DATA, EXPR_DATA_INITIAL

        variances = EXPR_DATA_INITIAL.var(axis=0)
        threshold = variances.quantile(min_var_quantile)
        keep = variances[variances >= threshold].index

        EXPR_DATA = EXPR_DATA_INITIAL[keep]

        return (
            f"Variance filtering applied.\n"
            f"- Quantile used: {min_var_quantile}\n"
            f"- Genes retained: {len(keep)}"
        )


    async def sample_outlier_scores() -> str:
        """
        Computes a simple sample-level outlier score based on
        average distance from the sample correlation centroid.

        Larger values indicate samples that are globally less
        correlated with the rest of the dataset.
        """
        sample_cor = EXPR_DATA.T.corr()
        dist = (1 - sample_cor).mean(axis=1)
        return dist.sort_values(ascending=False).to_string()


    async def compute_correlation(
        method: Annotated[
            Literal["pearson", "spearman"],
            Field(
                description=(
                    "Correlation method used to measure gene–gene similarity. "
                )
            ),
        ] = "pearson"
    ) -> str:
        """
        Computes the gene–gene correlation matrix from the
        filtered expression data.
        """
        RESULTS["cor_matrix"] = EXPR_DATA.corr(method=method)
        return f"{method.capitalize()} correlation matrix computed."


    def scale_free_fit_index(adj: np.ndarray,
                            n_bins: int = 20,
                            min_points: int = 3):
        """
        Compute the scale-free topology fit index (R^2) 
        following the logic of the original WGCNA implementation.

        Parameters
        ----------
        adj : np.ndarray
            Symmetric adjacency matrix.
        n_bins : int
            Number of bins in log10(k) space.
        min_points : int
            Minimum number of valid bins required for regression.

        Returns
        -------
        r2 : float
            Scale-free topology fit index (R^2).
        mean_k : float
            Mean network connectivity.
        """

        # ---- 1. Safety checks ----
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError("Adjacency matrix must be square.")

        # ---- 3. Compute connectivity ----
        k = np.sum(adj, axis=1) - 1

        k = np.maximum(k, 1e-10)

        mean_k = np.mean(k)

        # ---- 4. Log-transform connectivity ----
        log_k = np.log10(k)

        # ---- 5. Histogram in log space ----
        counts, bin_edges = np.histogram(log_k, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Keep only non-empty bins
        mask = counts > 0
        if np.sum(mask) < min_points:
            return np.nan, mean_k

        x = bin_centers[mask]
        y = np.log10(counts[mask] / np.sum(counts))

        # ---- 6. Linear regression in log–log space ----
        #slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        #r2 = r_value ** 2

        r_matrix = np.corrcoef(x, y)
        r2 = r_matrix[0, 1]**2

        return r2, mean_k



    async def pick_soft_threshold_power(
        network_type: Annotated[
            Literal["unsigned", "signed"],
            Field(
                description=(
                    "Type of co-expression network. "
                    "'unsigned' uses |correlation|^beta, "
                    "'signed' uses ((1 + correlation) / 2)^beta."
                )
            ),
        ] = "unsigned"
    ) -> str:
        """
        Evaluates candidate soft-thresholding powers by computing
        scale-free topology fit (R^2) and mean connectivity.

        This tool does not select a final beta automatically; it
        exposes the full diagnostic table so that the decision can
        be made externally.
        """
        cor = RESULTS.get("cor_matrix")
        powers = [int(p) for p in range(1,20)]

        rows = []
        for beta in powers:
            if network_type == "unsigned":
                adj = np.abs(cor.values) ** beta
            else:
                adj = ((1 + cor.values) / 2) ** beta

            r2, mean_k = scale_free_fit_index(adj)
            rows.append({
                "beta": beta,
                "scale_free_R2": r2,
                "mean_connectivity": mean_k
            })

        df = pd.DataFrame(rows)
        RESULTS["beta_table"] = df
        
        # Filtro de seguridad para evitar que el modelo elija ruido
        #filtered_df = df[df["scale_free_R2"] >= 0.75]
        filtered_df = df
        
        if filtered_df.empty:
            top_val = df.sort_values("scale_free_R2", ascending=False).head(3)
            return f"CRITICAL WARNING: No power reached R^2 >= 0.75. Data might be noisy. Best candidates:\n{top_val.to_string(index=False)}"
        
        return filtered_df.to_string(index=False)


    async def build_network_and_detect_modules(
        beta: Annotated[
            int,
            Field(description="Soft-thresholding power selected from the previous step.", gt=0)
        ],
        network_type: Annotated[
            Literal["unsigned", "signed"],
            Field(description="Network type used to compute adjacency.")
        ] = "unsigned",
        cut_height: Annotated[
            float,
            Field(
                description=(
                    "Tree cut height applied to TOM-based hierarchical clustering. "
                    "Typical values are between 0.90 and 0.99."
                ),
                gt=0.0,
                lt=1.0,
            ),
        ] = 0.95,
        min_module_size: Annotated[
            int,
            Field(
                description=(
                    "Minimum number of genes per module. "
                    "Smaller clusters are reassigned to the grey (0) module."
                ),
                gt=5,
            ),
        ] = 30
    ) -> str:
        """
        Builds the adjacency and TOM matrices and performs hierarchical
        clustering to identify co-expression modules.
        """
        cor = RESULTS["cor_matrix"]

        if network_type == "unsigned":
            adj = np.abs(cor.values) ** beta
        else:
            adj = ((1 + cor.values) / 2) ** beta

        np.fill_diagonal(adj, 0.0)

        k = adj.sum(axis=0)
        L = adj @ adj
        denom = np.minimum.outer(k, k) + 1 - adj
        tom = (L + adj) / np.maximum(denom, 1e-12)
        np.fill_diagonal(tom, 1.0)

        diss = 1 - tom
        Z = linkage(squareform(diss, checks=False), method="average")
        # Aplicamos el clustering
        clusters = fcluster(Z, t=cut_height, criterion="distance")
        labels = pd.Series(clusters, index=cor.index)
        
        # Consolidación: Tamaño mínimo estricto
        counts = labels.value_counts()
        small = counts[counts < min_module_size].index
        labels = labels.where(~labels.isin(small), other=0)
        
        RESULTS["labels"] = labels
        RESULTS["Z"] = Z
        RESULTS["beta"] = beta
        RESULTS["network_type"] = network_type
        RESULTS["cut_height"] = cut_height
        RESULTS["min_module_size"] = min_module_size

            
        return f"Module detection completed. Found {len(labels.unique())-1} modules. Grey genes: {counts.get(0, 0)}"


    async def correlate_modules_with_traits() -> str:
        """
        Computes module eigengenes (first principal component of each module)
        and correlates them with numeric sample traits using Pearson correlation.

        The grey module (0) is excluded from this analysis.
        """
        labels = RESULTS["labels"]
        traits = TRAIT_DATA

        results = []
        for mod in sorted(labels.unique()):
            if mod == 0:
                continue

            genes = labels[labels == mod].index
            X = EXPR_DATA[genes].values
            X -= X.mean(axis=0, keepdims=True)

            me = PCA(n_components=1).fit_transform(X).ravel()

            for tr in traits.columns:
                r, p = stats.pearsonr(me, traits[tr])
                results.append({
                    "module": mod,
                    "trait": tr,
                    "r": r,
                    "p": p
                })

        df = pd.DataFrame(results)
        RESULTS["module_trait_correlations"] = df

        return df.sort_values("p").to_string(index=False)
    '''

    url = "http://localhost:5001/update_notebook_blocks"

    payload = {
        "block_1": bloque1,

        "block_2": bloque2
    }

    try:
        requests.post(url, json=payload)
    except Exception as e:
        return f"❌ Connection error: {str(e)}"




if __name__ == "__main__":
    mcp.run()

