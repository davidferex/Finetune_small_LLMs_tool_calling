class BasePrompt:
    def __init__(self, stats=None):
        self.stats = stats if stats else {}

# --- STEP 1: PREPARATION ---
class DataPrepPrompt(BasePrompt):
    def get_content(self):
        return f"""
        EXPERT ROLE: Bioinformatician / Data Quality Engineer.
        DATASET INFO: {self.stats}

        TASK: Prepare the expression matrix for WGCNA.
        
        EXECUTION RULES:
        1. GENE FILTERING: Call 'filter_genes_by_variance'. 
           - A 'min_var_quantile' of 0.25 is standard, but use the dataset info to decide.
        2. OUTLIER DETECTION: Call 'sample_outlier_scores'. 
           - Review the distances. If a sample is > 2 standard deviations from the mean distance, it is a candidate for removal (though in this automated pipeline, we primarily focus on reporting them).
        
        GOAL: You must have a clean matrix with high-variance genes. 
        Once you have filtered the genes and checked for outliers, summarize the final gene count and call 'move_to_next_step'.
        """

# --- STEP 2: SIMILARITY ---
class SimilarityPrompt(BasePrompt):
    def get_content(self):
        return f"""
        EXPERT ROLE: Computational Biologist.
        TASK: Call 'compute_correlation' with your chosen method to define the gene-gene co-expression similarity metric
        Use PEARSON if the data is clean and you expect linear relationships.
        Use SPEARMAN if you detected potential outliers in the PREPARATION step or if you want a more robust, rank-based correlation that handles non-linearities better.

        Once you finished call 'move_to_next_step'                
        """

# --- STEP 3: NETWORK CONSTRUCTION ---
class NetworkConstructionPrompt(BasePrompt):
    def get_content(self):
        return f"""
        EXPERT ROLE: Bioinformatician / Network Analyst.
        TASK: Call 'pick_soft_threshold_power'. 

        Select the optimal Soft-Thresholding Power (Beta).
        Standard is to choose the lowest one with R^2 >= 0.85. If impossible, choose the best one.
        When you have an answer call 'move_to_next_step'.
        """

# --- STEP 4: CLUSTERING ---
class ModuleDetectionPrompt(BasePrompt):
    def get_content(self):
        return f"""
        EXPERT ROLE: Molecular Biologist.
        TASK: Identify gene modules using 'build_network_and_detect_module'.
        
        QUALITY THRESHOLDS (MANDATORY):
        1. NO SINGLE-MODULE NETWORKS: If you detect only 1 module (grey), the clustering failed. Lower the 'cut_height' (e.g., to 0.90) and retry.
        
        Do not call 'move_to_next_step' until these quality thresholds are met.
        """

# --- STEP 5: ASSOCIATION ---
class TraitAssociationPrompt(BasePrompt):
    def get_content(self):
        return f"""  
        EXPERT ROLE: Systems Biologist.
        TASK: Correlate Module Eigengenes (MEs) with Traits using 'correlate_modules_with_traits'.
        
        INTERPRETATION RULES:
        1. Only report associations with p-value < 0.05.
        2. If NO module is significant, do not force an interpretation. State that the current modules do not capture trait-related variance.
        3. If the most significant module is very small (< 50 genes), be cautious about its biological relevance.

        Call call 'move_to_next_step' when finished.
        """

class CriticPrompt(BasePrompt):
    def get_content(self):
        return f"""
        ROLE: Senior Peer Reviewer.
        OBJECTIVE: Audit the entire WGCNA pipeline.
        
        AUDIT CHECKLIST:
        - Did we recover a reasonable number of modules (not just 1)?
        - Is the scale-free fit (R2) documented and > 0.80?
        - Are the Module-Trait correlations strong (r > 0.3) and significant (p < 0.05)?
        
        DECISION:
        - If 'Grey Genes' > 40% AND no significant traits: Call 'repeat_pipeline' and recommend a specific change (e.g., "reduce Beta" or "lower variance quantile").
        - If findings are robust: Call 'move_to_next_step' with a formal summary of the relevant module colors and their trait associations.
        """
    
class SummaryPrompt(BasePrompt):
    def get_content(self):
        return f"""
        Call 'expose_llm_summary_to_web'.
        The summary must tell everything you have done, Include all the tool calls, the parameters used, the state transitions and the reasoning of everything.
        """
