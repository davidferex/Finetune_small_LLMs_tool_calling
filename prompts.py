class BasePrompt:
    def __init__(self, stats=None):
        self.stats = stats if stats else {}

# --- STEP 1: DATA CLEANING & PREPARATION (TAG: PREP) ---
class DataPrepPrompt(BasePrompt):
    def get_content(self):
        return f"""
        STEP 1: Data Cleaning & Initial Preparation.
        Dataset Context: {self.stats}

        TASKS:
        1. Remove lowly expressed genes.
        2. Handle missing values (NA/NaN) using appropriate imputation or removal.
        3. Detect sample outliers using PCA or Hierarchical Clustering tools.
        
        EXIT: Use 'move_to_next_step' when finished.
        """

# --- STEP 2: NORMALIZATION & TRANSFORMATION (TAG: NORM, TRANS) ---
class NormalizationPrompt(BasePrompt):
    def get_content(self):
        return f"""
        STEP 2: Normalization & Variance Stabilization.
        Platform: {self.stats.get('platform')} | Outlier Risk: {self.stats.get('outlier_risk')}
        
        GOAL: Make samples comparable and stabilize variance across the dynamic range.
        TASKS:
        1. Select the normalization method (TMM, DESeq2, CPM, etc.) best suited for {self.stats.get('platform')}.
        2. Apply a variance-stabilizing transformation (VST) or log2(x+1) to ensure homoscedasticity if needed.
        
        SUCCESS CRITERION: The data distribution should be suitable for correlation-based network analysis.
        EXIT: Use 'move_to_next_step' after the final transformation is applied.
        """

# --- STEP 3: SIMILARITY MEASUREMENT (TAG: SIMIL) ---
class SimilarityPrompt(BasePrompt):
    def get_content(self):
        return f"""
        STEP 3: Similarity Measurement.
        
        GOAL: Quantify relationships between gene pairs.
        CONTEXT: Based on Step 1, the outlier risk is {self.stats.get('outlier_risk')}.
        TASKS:
        1. Choose a correlation metric. If outliers remain a concern, prefer 'bicor' or 'Spearman'.
        
        EXIT: Use 'move_to_next_step' once the similarity matrix is generated.
        """

# --- STEP 4: NETWORK CONSTRUCTION ---
class NetworkConstructionPrompt(BasePrompt):
    def get_content(self):
        return f"""
        STEP 4: Network Construction.
        
        GOAL: Convert similarity into an adjacency matrix.
        TASKS:
        1. Use 'build_optimized_soft_threshold_network' to find the optimal beta power.
        2. Ensure the network follows a scale-free topology (R^2 > 0.85).
        
        EXIT: Use 'move_to_next_step' once the adjacency matrix is built.
        """

# --- STEP 5: REFINEMENT (TOM) ---
class NetworkRefinementPrompt(BasePrompt):
    def get_content(self):
        return f"""
        STEP 5: Network Topology Refinement (TOM).
        
        GOAL: Minimize noise and emphasize robust connections.
        TASKS:
        1. Calculate the Topological Overlap Matrix (TOM).
        
        EXIT: Use 'move_to_next_step' once the TOM-based dissimilarity matrix is ready.
        """

# --- STEP 6: MODULE DETECTION ---
class ModuleDetectionPrompt(BasePrompt):
    def get_content(self):
        return f"""
        STEP 6: Module (Cluster) Detection.
        
        GOAL: Group genes into tightly co-expressed modules.
        TASKS:
        1. Run hierarchical clustering on the TOM.
        2. Use 'apply_dynamic_tree_cut' to define the modules (colors).
        3. If the 'Grey' module is too large, try adjusting the 'deep_split' parameter.
        
        EXIT: Use 'move_to_next_step' after defining module assignments.
        """

# --- STEP 7: SUMMARIZATION ---
class ModuleSummarizationPrompt(BasePrompt):
    def get_content(self):
        return f"""
        STEP 7: Module Summarization.
        
        GOAL: Reduce modules to a representative signal.
        TASKS:
        1. Compute Module Eigengenes (ME).
        
        EXIT: Use 'move_to_next_step' once you have the [Samples x Modules] matrix.
        """

# --- STEP 8: ASSOCIATION ---
class TraitAssociationPrompt(BasePrompt):
    def get_content(self):
        return f"""
        STEP 8: Association with Traits.
        
        GOAL: Identify biologically relevant modules for the phenotype.
        TASKS:
        1. Correlate Eigengenes with clinical traits.
        2. Identify modules with significant p-values.
        3. Visualize results using a heatmap.
        
        EXIT: Use 'move_to_next_step' and summarize which module color is most significant.
        """

# --- STEP 9: HUB GENES ---
class HubGenePrompt(BasePrompt):
    def get_content(self):
        return f"""
        STEP 9: Hub Gene Identification.
        
        GOAL: Find key drivers within the significant modules.
        TASKS:
        1. Identify top hub genes based on connectivity (kIM) and module membership (kME).
        2. Validate if Hubs are also significant for the clinical trait.
        
        EXIT: Use 'move_to_next_step' after listing the candidate hub genes.
        """

# --- STEP 10: INTERPRETATION ---
class InterpretationPrompt(BasePrompt):
    def get_content(self):
        return f"""
        STEP 10: Functional Interpretation & Validation.
        
        GOAL: Translate gene lists into biological meaning.
        TASKS:
        1. Perform GO or Pathway enrichment analysis for the key modules.
        2. Compare with external datasets or suggest experimental validations.
        
        EXIT: Use 'move_to_next_step' with a FINAL REPORT summarizing the whole study.
        """