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
    #config = FastPrototypePreset.copy()
    #config["epochs"] = epochs
    #config["batch_size"] = batch_size
    
    # Simulación de ejecución:
    # gen = RealGenerator()
    # gen.generate(n_samples=n_samples, **config)
    
    return f"Success: Executed FastPrototypePreset for {n_samples} samples (Epochs: {epochs}, Batch size: {batch_size})."

@mcp.tool()
async def HighFidelityPreset(
    n_samples: Annotated[int, Field(description="Number of samples to generate")],
    auto_report: Annotated[bool, Field(description="Activate report or not")]
) -> str:
    """
    PRESET: HighFidelityPreset.
    DESCRIPTION: Tuned for maximum quality (CTGAN/TVAE with more training) for production data.
    Use this when the user requires high statistical accuracy and distribution matching for real-world scenarios.
    """
    #config = HighFidelityPreset.copy()
    #config["epochs"] = epochs
    # config["model"] = model_type
    
    return f"Success: Executed HighFidelityPreset for {n_samples} samples (Epochs: {epochs}, Model type: {model_type})."



@mcp.tool()
async def ImbalancedGeneratorPreset(
    n_samples: Annotated[int, Field(description="Total number of samples to generate after rebalancing")],
    target_column: Annotated[str, Field(description="The skewed column name to rebalance")],
    imbalance_ratio: Annotated[float, Field(description="Imbalance ratio for the minority class.")]
) -> str:
    """
    PRESET: ImbalancePreset.
    DESCRIPTION: Configured to handle and rebalance highly skewed datasets.
    Use this when the user mentions imbalanced classes, rare events, or the need to fix skewed data distributions.
    """
    # config = ImbalancePreset.copy()
    
    return f"Success: Executed ImbalancePreset targeting '{target_column}' with ratio {minority_class_ratio} (n_samples: {n_samples})."

@mcp.tool()
async def TimeSeriesPreset(
    n_samples: Annotated[int, Field(description="Number of sequences to generate")],
    sequence_key: Annotated[str, Field(description="Sequence key")],
    target_column: Annotated[str, Field(description="Target column of the series")],
    method: Annotated[str, Field(description="Method to use")]
) -> str:
    """
    PRESET: TimeSeriesPreset.
    DESCRIPTION: Setup for sequential data generation.
    Use this when the user describes temporal data, longitudinal studies, or patient monitoring over time.
    """
    # config = TimeSeriesPreset.copy()
    
    return f"Success: Executed TimeSeriesPreset. Sequences: {n_samples}, Length: {sequence_length}, Noise_std: {noise_std}."


@mcp.tool()
async def BalancedDataGeneratorPreset(
    n_samples: Annotated[int, Field(description="Number of synthetic samples to generate in order to balance the dataset")],
    target_col: Annotated[str, Field(description="Name of the target column that needs to be balanced")]
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

    # Aquí iría tu lógica real con SMOTE/ADASYN
    # Ejemplo conceptual:
    # balanced_df = apply_smote(original_df, target_col, n_samples)

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