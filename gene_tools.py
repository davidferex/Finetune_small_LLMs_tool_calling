import json
from fastmcp import FastMCP

mcp = FastMCP(name="Gene-CoExpression-WGCNA")

# --- ESTADO INTERNO PARA ACTUALIZAR STATS ---
DATASET_STATE = {
    "platform": "RNA-seq",
    "samples": 45,
    "genes_initial": 20000,
    "current_gene_count": 20000,
    "missing_values": 150, 
    "outlier_risk": "high",
    "phenotype_data": "available"
}

# --- DATASET STATISTICS (AHORA DINÁMICO) ---
@mcp.tool()
async def get_data_statistics() -> str:
    """Provides metadata and statistics for the currently loaded dataset."""
    return json.dumps(DATASET_STATE)

# --- STEP 1: INPUT DATA PREPARATION (TAG: PREP) ---
@mcp.tool(tags=["PREP"])
async def remove_lowly_expressed_genes(min_counts: int = 10) -> str:
    """
    Filters out lowly expressed genes that don't provide statistical signal. 
    Removes rows where the sum of counts is below the specified threshold.
    """
    DATASET_STATE["current_gene_count"] -= 4500 # Simulación de filtrado
    return f"Success: Filtering complete. Genes with fewer than {min_counts} total reads have been removed."

@mcp.tool(tags=["PREP"])
async def handle_missing_values(strategy: str = "impute") -> str:
    """
    Manages missing values (NA/NaN) in the matrix. 
    Strategies: 'remove' (delete row) or 'impute' (fill with mean/median).
    """
    DATASET_STATE["missing_values"] = 0 # Actualizamos el estado
    return f"Success: Missing values treated using strategy: {strategy}."

@mcp.tool(tags=["PREP"])
async def run_pca_outlier_detection() -> str:
    """
    Performs Principal Component Analysis (PCA) to identify 
    samples that behave as anomalies relative to the group.
    """
    DATASET_STATE["outlier_risk"] = "low" # El riesgo baja tras detectar/limpiar
    return "Result: PCA finished. No significant outliers detected in main clusters."

@mcp.tool(tags=["PREP"])
async def run_hierarchical_clustering_outliers() -> str:
    """
    Creates a sample dendrogram to detect outliers via hierarchical distance. 
    Standard WGCNA recommendation for initial data cleaning.
    """
    DATASET_STATE["outlier_risk"] = "low"
    return "Result: Hierarchical clustering complete. Sample dendrogram shows a clean structure."

# --- STEP 2: NORMALIZATION (TAG: NORM) ---

@mcp.tool(tags=["NORM"])
async def apply_tmm_normalization() -> str:
    """
    Trimmed Mean of M-values (TMM). 
    Standard RNA-seq method using scaling factors to normalize by library composition.
    """
    return "Success: TMM normalization complete. Scaling factors calculated."

@mcp.tool(tags=["NORM"])
async def apply_deseq2_size_factors() -> str:
    """
    DESeq2 Median of Ratios. 
    Calculates size factors based on geometric mean. Ideal for robust differential comparisons.
    """
    return "Success: DESeq2 size factors applied to the matrix."

@mcp.tool(tags=["NORM"])
async def apply_cpm_normalization() -> str:
    """
    Counts Per Million (CPM). 
    Simple normalization by sequencing depth. Does not correct for composition, only library size.
    """
    return "Success: Data transformed to CPM."

@mcp.tool(tags=["NORM"])
async def apply_rma_normalization() -> str:
    """
    Robust Multi-array Average (RMA). 
    Microarray specific. Includes background correction, quantile normalization, and summarization.
    """
    return "Success: RMA normalization finished for Microarray data."

@mcp.tool(tags=["NORM"])
async def apply_quantile_normalization() -> str:
    """
    Quantile Normalization. 
    Forces all samples to have the same intensity distribution. Common in Microarrays and Proteomics.
    """
    return "Success: Intensity distribution equalized via quantiles."

# --- STEP 2.2: TRANSFORMATIONS (TAG: TRANS) ---

@mcp.tool(tags=["NORM"])
async def apply_log2_plus_one() -> str:
    """
    Applies log2(x + 1) transformation. 
    Reduces data skewness and compresses dynamic range. The +1 prevents log of zero.
    """
    return "Success: log2(x+1) transformation complete."

@mcp.tool(tags=["NORM"])
async def apply_vst_transformation() -> str:
    """
    Variance Stabilizing Transformation (VST). 
    Elimina la dependencia de la varianza respecto a la media. Crucial para algoritmos que asumen homocedasticidad como WGCNA.
    """
    return "Success: Variance stabilized via VST. Matrix ready for network analysis."

# --- STEP 3: SIMILARITY MEASUREMENT (TAG: SIMIL) ---

@mcp.tool(tags=["SIMIL"])
async def compute_pearson_correlation() -> str:
    """
    Calculates Pearson correlation. Most common and fastest metric. 
    Ideal for normally distributed data without significant outliers.
    """
    return "Result: Similarity matrix (Pearson) generated. Dimensions: [Genes x Genes]."

@mcp.tool(tags=["SIMIL"])
async def compute_spearman_correlation() -> str:
    """
    Calculates Spearman correlation. 
    Non-parametric rank-based metric, highly robust against outliers and extreme values.
    """
    return "Result: Similarity matrix (Spearman) generated via rank transformation."

@mcp.tool(tags=["SIMIL"])
async def compute_biweight_midcorrelation() -> str:
    """
    Calculates Biweight Midcorrelation (bicor). 
    Recommended for WGCNA. Robust like Spearman but retains more signal intensity information. 
    Ideal if outliers were detected in STEP 1.
    """
    return "Result: Similarity matrix (bicor) generated. Metric optimized for WGCNA."

# --- STEP 4: NETWORK CONSTRUCTION (TAG: NETWORK) ---

@mcp.tool(tags=["NETWORK"])
async def build_optimized_soft_threshold_network() -> str:
    """
    Automatically analyzes network topology to find the optimal beta power 
    (Scale-free topology R^2 > 0.85) and builds the adjacency matrix in one step.
    """
    beta_selected = 12
    r_squared = 0.89
    return (f"Analysis & Construction Complete:\n"
            f"- Automatically selected power: β={beta_selected}\n"
            f"- Scale-free topology fit: R^2={r_squared}\n"
            f"- Status: Adjacency matrix generated.")

@mcp.tool(tags=["NETWORK"])
async def build_hard_threshold_adjacency(cutoff: float = 0.8) -> str:
    """
    Creates an adjacency matrix using Hard Thresholding. 
    Removes all connections with correlation below the specified cutoff.
    """
    return f"Success: Adjacency matrix generated with Hard Thresholding (Cutoff={cutoff})."

# --- STEP 5: NETWORK TOPOLOGY REFINEMENT (TAG: TOPO) ---

@mcp.tool(tags=["TOPO"])
async def compute_topological_overlap_matrix() -> str:
    """
    Transforms adjacency matrix into a Topological Overlap Matrix (TOM). 
    Filters spurious connections by emphasizing shared neighbors.
    """
    return ("Success: TOM transformation complete.\n"
            "- Background noise reduced via shared neighbor analysis.\n"
            "- Dissimilarity matrix generated for clustering.")

# --- STEP 6: MODULE DETECTION (TAG: CLUSTERING) ---

@mcp.tool(tags=["CLUSTERING"])
async def run_hierarchical_clustering_on_tom() -> str:
    """
    Performs hierarchical clustering based on the TOM dissimilarity matrix. 
    Generates the gene dendrogram required for module identification.
    """
    return "Result: Gene dendrogram generated. Cluster structure ready for cutting."

@mcp.tool(tags=["CLUSTERING"])
async def apply_dynamic_tree_cut(min_module_size: int = 30, deep_split: int = 2) -> str:
    """
    Applies the Dynamic Tree Cut algorithm to identify gene modules (colors).
    - min_module_size: Minimum number of genes per group.
    - deep_split: Cutting sensitivity (0-4). Higher values result in more small modules.
    """
    return (f"Success: Module identification complete.\n"
            f"- Modules detected: 12 (Turquoise, Blue, Brown, etc.).\n"
            f"- Grey module size (unassigned genes): 450.\n"
            f"- Parameters used: minSize={min_module_size}, deepSplit={deep_split}.")

# --- STEP 7: MODULE SUMMARIZATION (TAG: SUMMARY) ---

@mcp.tool(tags=["SUMMARY"])
async def compute_module_eigengenes() -> str:
    """
    Calculates the Module Eigengene (ME) for each identified color. 
    Representa el primer componente principal de la expresión de un módulo y 
    sirve como perfil de expresión característico del grupo de genes.
    """
    return ("Success: Eigengenes calculated for all modules.\n"
            "- Matrix [Samples x Modules] generated.\n"
            "- Ready for clinical trait correlation.")

# --- STEP 8: ASSOCIATION WITH TRAITS (TAG: ASSOCIATION) ---

@mcp.tool(tags=["ASSOCIATION"])
async def correlate_modules_with_clinical_traits() -> str:
    """
    Calculates correlation (Pearson or bicor) between Module Eigengenes and 
    clinical traits/phenotypes. Generates a correlation matrix and p-values.
    """
    return ("Success: Module-Trait correlation complete.\n"
            "- Identified 3 significant modules for 'Disease_Status'.\n"
            "- Key Modules: Turquoise (r=0.82), Brown (r=-0.65), Green (r=0.45).\n"
            "- Results saved to 'trait_correlation_matrix.csv'.")

@mcp.tool(tags=["ASSOCIATION"])
async def visualize_module_trait_heatmap() -> str:
    """
    Generates a Heatmap visualizing correlations between modules and traits, 
    annotating relationship strength and statistical significance.
    """
    return "Success: Heatmap generated as 'module_trait_heatmap.pdf'."

# --- STEP 9: HUB GENE IDENTIFICATION (TAG: HUB) ---

@mcp.tool(tags=["HUB"])
async def identify_top_hub_genes(module_color: str, top_n: int = 10) -> str:
    """
    Calculates intramodular connectivity (kIM) and module membership (kME). 
    Identifies genes with the highest centrality in a specific module.
    """
    return (f"Success: Top {top_n} Hub Genes identified for the {module_color} module.\n"
            f"- Key Hubs: [Gene_A, Gene_B, Gene_C...]\n"
            f"- These genes are primary candidates for experimental drivers.")

@mcp.tool(tags=["HUB"])
async def calculate_gene_significance_vs_kme(module_color: str, trait_name: str) -> str:
    """
    Crosses Gene Significance for a trait with module membership (kME). 
    Genes scoring high in both are the most reliable hubs.
    """
    return (f"Success: GS vs kME analysis complete for {module_color}.\n"
            f"- Strong correlation (r=0.8) found between gene importance and centrality.\n"
            f"- Confirms hubs are relevant for {trait_name}.")

# --- STEP 10: FUNCTIONAL INTERPRETATION (TAG: INTERPRET) ---

@mcp.tool(tags=["INTERPRET"])
async def run_pathway_enrichment_analysis(module_color: str, database: str = "GO_BP") -> str:
    """
    Performs functional enrichment analysis (Gene Ontology or KEGG). 
    Identifies biological processes overrepresented in the module's gene list.
    Databases: 'GO_BP', 'KEGG', 'Reactome'.
    """
    return (f"Success: Enrichment for {module_color} complete.\n"
            f"- Top Term: 'Cell Cycle Regulation' (p-adj: 1e-12).\n"
            f"- Additional Processes: 'DNA Repair', 'Mitosis'.")

@mcp.tool(tags=["INTERPRET"])
async def compare_with_external_datasets(dataset_id: str = "GSE12345") -> str:
    """
    Compares identified modules and hubs with external published studies 
    to validate if the molecular signature is reproducible.
    """
    return (f"Success: External validation against {dataset_id} complete.\n"
            f"- 85% of Hub genes show consistent direction of change.\n"
            f"- High module preservation (Z-summary > 10).")

@mcp.tool(tags=["INTERPRET"])
async def suggest_experimental_validation() -> str:
    """
    Proposes a list of lab experiments (qPCR, CRISPR) based on the most 
    promising Hub genes to validate findings in vitro/in vivo.
    """
    return "Success: Validation plan generated. Priority: Knockdown of Top Hubs."

# --- FLOW CONTROL ---
@mcp.tool(tags=["MOVE"])
async def move_to_next_step(summary: str) -> str:
    """
    Finalizes the current state and moves to the next node in the pipeline. 
    YOU MUST provide a brief summary of your findings/actions in this step.
    """
    return f"TRANSITION: Step completed. Summary: {summary}"

if __name__ == "__main__":
    mcp.run()