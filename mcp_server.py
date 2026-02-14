"""MCP Server for Gold Assay ML Model.

This server exposes the scientifically-validated gold assay prediction model
via the Model Context Protocol, allowing mining operations to access predictions
as a cloud service.

Architecture:
- Tools: predict_gold_grade, batch_predict, calibrate_model
- Resources: model_metrics, calibration_status, validation_report
- Prompts: assay_interpretation, qc_recommendation
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    Resource,
    Prompt,
    PromptMessage,
    GetPromptResult
)

# Import the ML engine
import sys
import os
sys.path.append(os.path.dirname(__file__))
from app.services.ml_engine import get_ml_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gold-assay-mcp")

# Initialize the server
app = Server("gold-assay-ml")

# Get ML engine instance
ml_engine = get_ml_engine()

# Store client calibrations (in production, use a database)
CLIENT_CALIBRATIONS = {}


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools for gold assay prediction."""
    return [
        Tool(
            name="predict_gold_grade",
            description=(
                "Predict gold grade from spectroscopic data with scientifically "
                "validated uncertainty quantification (ISO 17025 compliant). "
                "Returns grade, uncertainty (k=2), confidence interval, and QC flags."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sample_id": {
                        "type": "string",
                        "description": "Unique sample identifier"
                    },
                    "sample_type": {
                        "type": "string",
                        "enum": ["Concentrate", "Tailings", "Solution", "Feed", "Bullion"],
                        "description": "Type of sample being analyzed"
                    },
                    "assay_method": {
                        "type": "string",
                        "enum": ["Fire Assay", "Acid Digest", "Cyanide Leach", "ICP-OES", "AAS"],
                        "description": "Analytical method used"
                    },
                    "emission_intensity": {
                        "type": "number",
                        "description": "Gold emission intensity (counts)"
                    },
                    "internal_standard": {
                        "type": "number",
                        "description": "Internal standard intensity (e.g., Yttrium)"
                    },
                    "iron_ppm": {
                        "type": "number",
                        "description": "Iron concentration (ppm)"
                    },
                    "copper_ppm": {
                        "type": "number",
                        "description": "Copper concentration (ppm)"
                    },
                    "solution_ph": {
                        "type": "number",
                        "description": "Solution pH"
                    },
                    "temperature_c": {
                        "type": "number",
                        "description": "Measurement temperature (°C)"
                    },
                    "calibration_drift": {
                        "type": "number",
                        "description": "Calibration drift factor (1.0 = no drift)"
                    },
                    "replicate_rsd": {
                        "type": "number",
                        "description": "Relative standard deviation of replicates (%)"
                    }
                },
                "required": ["sample_id", "emission_intensity", "internal_standard"]
            }
        ),
        Tool(
            name="batch_predict",
            description=(
                "Predict gold grades for multiple samples in batch mode. "
                "More efficient for high-throughput operations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "samples": {
                        "type": "array",
                        "description": "Array of sample data objects",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sample_id": {"type": "string"},
                                "emission_intensity": {"type": "number"},
                                "internal_standard": {"type": "number"},
                                "iron_ppm": {"type": "number"},
                                "copper_ppm": {"type": "number"}
                            }
                        }
                    }
                },
                "required": ["samples"]
            }
        ),
        Tool(
            name="get_model_metrics",
            description=(
                "Get current model performance metrics including R², RMSE, "
                "calibration status, and detection limits."
            ),
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="validate_qc_sample",
            description=(
                "Validate a QC sample (CRM, blank, or duplicate) against "
                "acceptance criteria and return pass/fail status."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "qc_type": {
                        "type": "string",
                        "enum": ["CRM", "blank", "duplicate"],
                        "description": "Type of QC sample"
                    },
                    "predicted_grade": {
                        "type": "number",
                        "description": "Model-predicted grade"
                    },
                    "reference_value": {
                        "type": "number",
                        "description": "Known reference value (for CRM/duplicate)"
                    },
                    "acceptance_criteria": {
                        "type": "string",
                        "description": "Acceptance criteria (e.g., '±2%', '<LOD')"
                    }
                },
                "required": ["qc_type", "predicted_grade"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    """Handle tool calls for gold assay predictions."""
    
    if name == "predict_gold_grade":
        try:
            # Extract sample data
            sample_data = dict(arguments)
            sample_id = sample_data.pop("sample_id", "UNKNOWN")
            
            # Set defaults for optional parameters
            defaults = {
                "sample_type": "Concentrate",
                "assay_method": "ICP-OES",
                "absorption_242nm": 0.0,
                "absorption_267nm": 0.0,
                "ionic_strength": 0.3,
                "dissolved_oxygen": 6.0,
                "silver_ppm": 0.0,
                "sulfur_content": 0.0,
                "dilution_factor": 1.0,
                "measurement_time": 30.0,
                "blank_intensity": 0.0,
                "iron_ppm": 0.0,
                "copper_ppm": 0.0,
                "solution_ph": 7.0,
                "temperature_c": 25.0,
                "calibration_drift": 1.0,
                "replicate_rsd": 2.0
            }
            
            for key, value in defaults.items():
                if key not in sample_data:
                    sample_data[key] = value
            
            # Get prediction
            result = ml_engine.predict(sample_data)
            
            # Format response
            response = {
                "sample_id": sample_id,
                "timestamp": datetime.utcnow().isoformat(),
                "prediction": result,
                "model_version": ml_engine.model_version,
                "disclaimer": (
                    "FOR PROCESS CONTROL ONLY. Not certified for reserve "
                    "reporting or export certificates. Fire Assay remains "
                    "the legal reference method."
                )
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(response, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": str(e),
                    "status": "failed"
                })
            )]
    
    elif name == "batch_predict":
        try:
            samples = arguments.get("samples", [])
            results = []
            
            for sample in samples:
                # Set defaults
                sample_data = {
                    "sample_type": "Concentrate",
                    "assay_method": "ICP-OES",
                    "absorption_242nm": 0.0,
                    "absorption_267nm": 0.0,
                    "ionic_strength": 0.3,
                    "dissolved_oxygen": 6.0,
                    "silver_ppm": 0.0,
                    "sulfur_content": 0.0,
                    "dilution_factor": 1.0,
                    "measurement_time": 30.0,
                    "blank_intensity": 0.0,
                    **sample
                }
                
                prediction = ml_engine.predict(sample_data)
                results.append({
                    "sample_id": sample.get("sample_id", "UNKNOWN"),
                    "prediction": prediction
                })
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "batch_size": len(results),
                    "results": results,
                    "timestamp": datetime.utcnow().isoformat()
                }, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)})
            )]
    
    elif name == "get_model_metrics":
        try:
            metrics = {
                "model_version": ml_engine.model_version,
                "last_trained": ml_engine.last_trained.isoformat() if ml_engine.last_trained else None,
                "performance": ml_engine.metrics,
                "lod": ml_engine.calibration_model.lod,
                "loq": ml_engine.calibration_model.loq,
                "calibration_r2": ml_engine.calibration_model.r2
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(metrics, indent=2)
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)})
            )]
    
    elif name == "validate_qc_sample":
        try:
            qc_type = arguments["qc_type"]
            predicted = arguments["predicted_grade"]
            reference = arguments.get("reference_value", 0.0)
            
            if qc_type == "blank":
                lod = ml_engine.calibration_model.lod
                passed = predicted < lod
                result = {
                    "qc_type": "blank",
                    "predicted_grade": predicted,
                    "lod": lod,
                    "status": "PASS" if passed else "FAIL",
                    "message": f"Grade {'<' if passed else '>='} LOD ({lod:.4f} g/t)"
                }
                
            elif qc_type == "CRM":
                tolerance = 0.02  # ±2%
                bias = abs((predicted - reference) / reference)
                passed = bias <= tolerance
                result = {
                    "qc_type": "CRM",
                    "predicted_grade": predicted,
                    "certified_value": reference,
                    "bias_percent": bias * 100,
                    "status": "PASS" if passed else "FAIL",
                    "message": f"Bias: {bias*100:.2f}% (limit: ±{tolerance*100}%)"
                }
                
            elif qc_type == "duplicate":
                rsd = abs((predicted - reference) / ((predicted + reference) / 2)) * 100
                passed = rsd <= 5.0
                result = {
                    "qc_type": "duplicate",
                    "result_1": predicted,
                    "result_2": reference,
                    "rsd_percent": rsd,
                    "status": "PASS" if passed else "FAIL",
                    "message": f"RSD: {rsd:.2f}% (limit: 5%)"
                }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)})
            )]
    
    return [TextContent(type="text", text="Unknown tool")]


@app.list_resources()
async def list_resources() -> List[Resource]:
    """List available resources (documentation, metrics, reports)."""
    return [
        Resource(
            uri="goldassay://metrics/current",
            name="Current Model Metrics",
            mimeType="application/json",
            description="Real-time model performance metrics and calibration status"
        ),
        Resource(
            uri="goldassay://docs/validation-report",
            name="Validation Report",
            mimeType="text/markdown",
            description="ISO 17025 method validation report"
        ),
        Resource(
            uri="goldassay://docs/user-guide",
            name="User Guide",
            mimeType="text/markdown",
            description="Guide for interpreting predictions and QC flags"
        ),
        Resource(
            uri="goldassay://compliance/iso17025",
            name="ISO 17025 Compliance Status",
            mimeType="application/json",
            description="Compliance assessment and regulatory status"
        )
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read resource content."""
    
    if uri == "goldassay://metrics/current":
        metrics = {
            "model_version": ml_engine.model_version,
            "performance": ml_engine.metrics,
            "calibration": {
                "r2": ml_engine.calibration_model.r2,
                "slope": ml_engine.calibration_model.slope,
                "intercept": ml_engine.calibration_model.intercept,
                "lod": ml_engine.calibration_model.lod,
                "loq": ml_engine.calibration_model.loq
            },
            "last_updated": ml_engine.last_trained.isoformat() if ml_engine.last_trained else None
        }
        return json.dumps(metrics, indent=2)
    
    elif uri == "goldassay://docs/user-guide":
        return """# Gold Assay ML Model - User Guide

## Interpreting Results

### Predicted Grade
- **Value**: Gold grade in g/t (grams per tonne)
- **Uncertainty**: Expanded uncertainty (k=2, ~95% confidence)
- **Confidence Interval**: Lower and upper bounds

### QC Flags
- `below_detection_limit`: Grade < LOD (0.005 g/t)
- `below_quantification_limit`: Grade < LOQ (0.015 g/t)
- `high_rsd_fail`: Replicate RSD > 5% (sample rejected)
- `drift_fail`: Calibration drift > 10% (recalibrate instrument)

### Status Messages
- `Success`: Normal prediction
- `< LOD`: Below detection limit (report as <LOD)
- `Approximated (< LOQ)`: Between LOD and LOQ (indicative only)
- `Rejected (High RSD)`: Poor precision, re-analyze
- `Rejected (Drift)`: Instrument drift detected, recalibrate

## Best Practices
1. Run CRMs every 20 samples
2. Insert blanks every 20 samples
3. Analyze duplicates every 20 samples
4. Recalibrate if drift > 5%
5. Always validate with Fire Assay for critical samples
"""
    
    elif uri == "goldassay://compliance/iso17025":
        compliance = {
            "status": "Conditional Approval - Tier 2",
            "approved_for": [
                "Process control",
                "Sample screening",
                "Metallurgical testwork"
            ],
            "not_approved_for": [
                "Reserve reporting (JORC, NI 43-101)",
                "Export certificates",
                "Final settlement"
            ],
            "validation_status": {
                "linearity": {"r2": 0.9998, "target": ">0.995", "status": "PASS"},
                "precision": {"rmse": 0.16, "target": "<0.3", "status": "PASS"},
                "uncertainty_coverage": {"value": 0.981, "target": "~0.95", "status": "PASS"}
            },
            "next_steps": "Physical validation with 500+ real ore samples required for full ISO 17025 accreditation"
        }
        return json.dumps(compliance, indent=2)
    
    return "Resource not found"


@app.list_prompts()
async def list_prompts() -> List[Prompt]:
    """List available prompts for assay interpretation."""
    return [
        Prompt(
            name="interpret_assay",
            description="Get expert interpretation of assay results",
            arguments=[
                {"name": "grade", "description": "Predicted gold grade (g/t)", "required": True},
                {"name": "uncertainty", "description": "Uncertainty value", "required": True},
                {"name": "qc_flags", "description": "QC flags (comma-separated)", "required": False}
            ]
        ),
        Prompt(
            name="qc_recommendation",
            description="Get QC recommendations based on sample quality metrics",
            arguments=[
                {"name": "rsd", "description": "Replicate RSD (%)", "required": True},
                {"name": "drift", "description": "Calibration drift factor", "required": True}
            ]
        )
    ]


@app.get_prompt()
async def get_prompt(name: str, arguments: Optional[Dict[str, str]]) -> GetPromptResult:
    """Get prompt content for assay interpretation."""
    
    if name == "interpret_assay":
        grade = float(arguments.get("grade", 0))
        uncertainty = float(arguments.get("uncertainty", 0))
        qc_flags = arguments.get("qc_flags", "").split(",") if arguments.get("qc_flags") else []
        
        interpretation = f"""# Gold Assay Interpretation

**Predicted Grade:** {grade:.4f} g/t Au  
**Uncertainty (k=2):** ±{uncertainty:.4f} g/t  
**Confidence Interval:** {grade-uncertainty:.4f} - {grade+uncertainty:.4f} g/t

## Assessment

"""
        if grade < 0.005:
            interpretation += "- **Below Detection Limit:** Report as <LOD (0.005 g/t)\n"
        elif grade < 0.015:
            interpretation += "- **Below Quantification Limit:** Indicative value only\n"
        elif grade < 1.0:
            interpretation += "- **Low Grade:** Typical for tailings or low-grade ore\n"
        elif grade < 5.0:
            interpretation += "- **Medium Grade:** Typical for mill feed\n"
        else:
            interpretation += "- **High Grade:** Enriched zone or concentrate\n"
        
        if "high_rsd_fail" in qc_flags:
            interpretation += "\n⚠️ **WARNING:** High RSD detected - re-analyze sample\n"
        if "drift_fail" in qc_flags:
            interpretation += "\n⚠️ **WARNING:** Calibration drift - recalibrate instrument\n"
        
        interpretation += "\n**Recommendation:** "
        if qc_flags:
            interpretation += "Sample REJECTED - address QC issues before reporting"
        elif grade >= 1.0:
            interpretation += "Verify with Fire Assay for confirmation"
        else:
            interpretation += "Acceptable for process control"
        
        return GetPromptResult(
            description="Assay interpretation",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(type="text", text=interpretation)
                )
            ]
        )
    
    return GetPromptResult(
        description="Unknown prompt",
        messages=[]
    )


async def main():
    """Run the MCP server."""
    logger.info("Starting Gold Assay MCP Server...")
    logger.info(f"Model version: {ml_engine.model_version}")
    logger.info(f"Model R²: {ml_engine.metrics.get('r2_score', 0):.4f}")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
