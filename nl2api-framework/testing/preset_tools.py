import pandas as pd
from typing import Annotated, Optional, Literal
from pydantic import Field
from fastmcp import FastMCP
from calm_data_generator import RealGenerator
# from calm_data_generator.presets import FastPrototypePreset
# from calm_data_generator.presets import HighFidelityPreset
# from calm_data_generator.presets import ImbalancePreset
# from calm_data_generator.presets import TimeSeriesPreset


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