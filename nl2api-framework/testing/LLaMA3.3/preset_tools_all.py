import pandas as pd
from typing import Annotated, Optional, Literal, List
from pydantic import Field
from fastmcp import FastMCP
from calm_data_generator import RealGenerator


mcp = FastMCP(name="Presets-MCP-Server")

# --- MCP TOOLS: GENERACIÓN POR PRESETS ---

@mcp.tool()
async def FastPrototypePreset(
    n_samples: Annotated[int, Field(description="Number of samples to generate")]
) -> str:
    """
    PRESET: FastPrototypePreset.
    DESCRIPTION: Optimized for speed (fewer epochs, simple models) to test pipelines quickly.
    Use this when the user needs to iterate fast or verify the workflow integrity without waiting for high-quality training.
    """
    return f"Success: Executed FastPrototypePreset for {n_samples} samples."

@mcp.tool()
async def HighFidelityPreset(
    n_samples: Annotated[int, Field(description="Number of samples to generate")],
    auto_report: Annotated[Optional[bool], Field(default=True, description="Whether to automatically generate a validation report after generation")] = True
) -> str:
    """
    PRESET: HighFidelityPreset.
    DESCRIPTION: Tuned for maximum quality (CTGAN/TVAE with more training) for production data.
    Use this when the user requires high statistical accuracy and distribution matching for real-world scenarios.
    """
    return f"Success: Executed HighFidelityPreset for {n_samples} samples (auto_report={auto_report})."


@mcp.tool()
async def ImbalancedGeneratorPreset(
    n_samples: Annotated[int, Field(description="Number of samples to generate")],
    target_col: Annotated[str, Field(description="Name of the target column to apply the imbalanced distribution to")],
    imbalance_ratio: Annotated[Optional[float], Field(default=0.1, description="Ratio of the minority class (e.g. 0.1 means 10% minority, 90% majority)")] = 0.1
) -> str:
    """
    PRESET: ImbalancePreset.
    DESCRIPTION: Configured to handle and rebalance highly skewed datasets.
    Use this when the user mentions imbalanced classes, rare events, or the need to fix skewed data distributions.
    """
    return f"Success: Executed ImbalancePreset targeting '{target_col}' with ratio {imbalance_ratio} (n_samples: {n_samples})."

@mcp.tool()
async def TimeSeriesPreset(
    n_samples: Annotated[int, Field(description="Number of samples to generate")],
    sequence_key: Annotated[str, Field(description="Column used to group rows into individual sequences")],
    time_key: Annotated[Optional[str], Field(default=None, description="Name of the column containing the timestamp or date")] = None,
    method: Annotated[Optional[Literal["timegan", "timevae", "fflows"]], Field(default="fflows", description="Generation method: 'timegan' for complex patterns, 'timevae' for speed, 'fflows' for periodic/seasonal series")] = "fflows"
) -> str:
    """
    PRESET: TimeSeriesPreset.
    DESCRIPTION: Setup for sequential data generation.
    Use this when the user describes temporal data, longitudinal studies, or patient monitoring over time.
    """
    return f"Success: Executed TimeSeriesPreset. n_samples={n_samples}, sequence_key={sequence_key}, time_key={time_key}, method={method}."


@mcp.tool()
async def BalancedDataGeneratorPreset(
    n_samples: Annotated[int, Field(description="Number of samples to generate")],
    target_col: Annotated[str, Field(description="Name of the target column whose class distribution will be balanced")]
) -> str:
    """
    PRESET: BalancedDataGeneratorPreset.
    DESCRIPTION: Preset designed to balance an originally imbalanced dataset.
    Uses SMOTE (or ADASYN) to oversample minority classes to achieve a balanced distribution.

    Use this when the user mentions:
    - balancing a dataset
    - fixing class imbalance
    - oversampling minority class
    - equalizing class distribution
    """

    return (
        f"Success: Executed BalancedDataGeneratorPreset. "
        f"Added {n_samples} synthetic samples to balance column '{target_col}'."
    )

@mcp.tool()
async def MoreInformationNeeded(
    missing: Annotated[list[str],Field(description="List of missing required argument names that must be provided before calling a preset")]
) -> str:
    """
    PRESET: MoreInformationNeeded.
    DESCRIPTION: Indicates that the user query is missing required arguments.
    Returns a list of the missing arguments that must be provided before a preset can be called.

    Use this when:
    - The user request does not specify all mandatory parameters
    - A preset cannot be safely executed due to incomplete information
    - Critical configuration values are absent
    """

    return (
        "More information required. Missing required arguments: "
        f"{', '.join(missing)}."
    )

@mcp.tool()
async def ConceptDriftPreset(
    n_samples: Annotated[int, Field(description="Number of samples to generate")],
    target_col: Annotated[str, Field(description="Name of the target column where the drift is applied")]
) -> str:
    """
    PRESET: ConceptDriftPreset.
    DESCRIPTION: Simulates sudden concept drift by altering the relationship between features and target.
    Use this when testing model robustness to changes in P(y|x).
    """
    return (
        f"Success: Executed ConceptDriftPreset. "
        f"Applied concept drift to '{target_col}' with {n_samples} samples."
    )

@mcp.tool()
async def DiffusionPreset(
    n_samples: Annotated[int, Field(description="Number of samples to generate")]
) -> str:
    """
    PRESET: DiffusionPreset.
    DESCRIPTION: Uses Tabular Diffusion Models (TabDDPM) for high-quality synthetic data generation.
    """
    return (
        f"Success: Executed DiffusionPreset using TabDDPM for {n_samples} samples."
    )

@mcp.tool()
async def RareDiseasePreset(
    n_samples: Annotated[int, Field(description="Number of samples to generate")],
    disease_ratio: Annotated[Optional[float], Field(default=0.01, description="Ratio of the rare disease in the data")] = 0.01
) -> str:
    """
    PRESET: RareDiseasePreset.
    DESCRIPTION: Simulates a clinical dataset with extremely low disease prevalence.
    """
    return (
        f"Success: Executed RareDiseasePreset with disease_ratio={disease_ratio} "
        f"for {n_samples} samples."
    )

@mcp.tool()
async def LongitudinalHealthPreset(
    n_samples: Annotated[int, Field(description="Number of samples to generate")],
    n_visits: Annotated[Optional[int], Field(default=5, description="Average number of visits per patient")] = 5
) -> str:
    """
    PRESET: LongitudinalHealthPreset.
    DESCRIPTION: Generates longitudinal clinical data with multiple patient visits over time.
    """
    return (
        f"Success: Executed LongitudinalHealthPreset with {n_visits} visits "
        f"for {n_samples} samples."
    )

@mcp.tool()
async def SeasonalTimeSeriesPreset(
    n_samples: Annotated[int, Field(description="Number of samples to generate")],
    time_col: Annotated[str, Field(description="Column with the timestamp")],
    seasonal_cols: Annotated[List[str], Field(description="List of columns to inject seasonality into")],
    period: Annotated[Optional[int], Field(default=12, description="Seasonality period")] = 12,
    amplitude: Annotated[Optional[float], Field(default=1.0, description="Strength of seasonal effect")] = 1.0
) -> str:
    """
    PRESET: SeasonalTimeSeriesPreset.
    DESCRIPTION: Generates time-series data with sinusoidal seasonal patterns.
    """
    return (
        f"Success: Executed SeasonalTimeSeriesPreset with period={period}, "
        f"amplitude={amplitude}, time_col={time_col}, seasonal_cols={seasonal_cols}, "
        f"n_samples={n_samples}."
    )


# --- NEW TOOLS ---

@mcp.tool()
async def CopulaPreset(
    n_samples: Annotated[int, Field(description="Number of samples to generate")]
) -> str:
    """
    PRESET: CopulaPreset.
    DESCRIPTION: Uses Gaussian Copula to model dependencies. Very fast and statistically robust baseline, though supports privacy less than GANs.
    """
    return f"Success: Executed CopulaPreset for {n_samples} samples."


@mcp.tool()
async def DataQualityAuditPreset(
    n_samples: Annotated[int, Field(description="Number of samples to generate")]
) -> str:
    """
    PRESET: DataQualityAuditPreset.
    DESCRIPTION: Focused on high-integrity generation with comprehensive automated reporting. Uses TVAE (often better than CTGAN for structure) and enables full reporting.
    """
    return f"Success: Executed DataQualityAuditPreset for {n_samples} samples."


@mcp.tool()
async def DriftScenarioPreset(
    n_samples: Annotated[int, Field(description="Number of samples to generate")],
    drift_scenarios: Annotated[Optional[List[dict]], Field(default=None, description="List with dictionaries defining drift scenarios.")] = None
) -> str:
    """
    PRESET: DriftScenarioPreset.
    DESCRIPTION: Preset designed to generate data with specific injected drift characteristics. Used for stress-testing ML models and drift detection systems.
    """
    return (
        f"Success: Executed DriftScenarioPreset for {n_samples} samples "
        f"(drift_scenarios={drift_scenarios})."
    )


@mcp.tool()
async def FastPreset(
    n_samples: Annotated[int, Field(description="Number of samples to generate")]
) -> str:
    """
    PRESET: FastPreset.
    DESCRIPTION: Preset designed to generate data as quickly as possible, sacrificing some quality. Uses LightGBM for speed, minimal reporting.
    """
    return f"Success: Executed FastPreset for {n_samples} samples."


@mcp.tool()
async def GradualDriftPreset(
    n_samples: Annotated[int, Field(description="Number of samples to generate")],
    drift_cols: Annotated[List[str], Field(description="List of columns to apply gradual drift to")],
    slope: Annotated[Optional[float], Field(default=0.01, description="The rate of drift (e.g., 0.01 means a 1% change per time unit)")] = 0.01
) -> str:
    """
    PRESET: GradualDriftPreset.
    DESCRIPTION: Simulates gradual drift over time or index.
    """
    return (
        f"Success: Executed GradualDriftPreset for {n_samples} samples "
        f"(drift_cols={drift_cols}, slope={slope})."
    )


@mcp.tool()
async def OmicsIntegrationPreset(
    n_samples: Annotated[int, Field(description="Number of samples to generate")],
    n_genes: Annotated[Optional[int], Field(default=100, description="Number of gene expression features to generate")] = 100,
    n_proteins: Annotated[Optional[int], Field(default=50, description="Number of proteomics features to generate")] = 50
) -> str:
    """
    PRESET: OmicsIntegrationPreset.
    DESCRIPTION: Generates multi-omics data (Clinical + Gene Expression + Proteomics) with high correlation integrity between layers.
    """
    return (
        f"Success: Executed OmicsIntegrationPreset for {n_samples} samples "
        f"(n_genes={n_genes}, n_proteins={n_proteins})."
    )


@mcp.tool()
async def SingleCellQualityPreset(
    n_samples: Annotated[int, Field(description="Number of samples to generate")]
) -> str:
    """
    PRESET: SingleCellQualityPreset.
    DESCRIPTION: Preset designed to generate high-quality single-cell RNA-seq data.
    """
    return f"Success: Executed SingleCellQualityPreset for {n_samples} samples."