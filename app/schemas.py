"""Pydantic schemas for data validation."""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from datetime import datetime


class SampleInput(BaseModel):
    """Input schema for sample analysis."""
    sample_type: str = Field(default='Concentrate', 
                            pattern='^(Concentrate|Tailings|Solution|Feed|Bullion)$')
    assay_method: str = Field(default='Fire Assay',
                             pattern='^(Fire Assay|Acid Digest|Cyanide Leach|ICP-OES|AAS)$')
    
    # Spectroscopic measurements
    absorption_242nm: float = Field(default=0.0, ge=0)
    absorption_267nm: float = Field(default=0.0, ge=0)
    emission_intensity: float = Field(default=0.0, ge=0)
    
    # Solution parameters
    solution_ph: float = Field(default=7.0, ge=0, le=14)
    temperature_c: float = Field(default=25.0, ge=0, le=100)
    ionic_strength: float = Field(default=0.0, ge=0)
    dissolved_oxygen: float = Field(default=0.0, ge=0)
    
    # Interference elements
    iron_ppm: float = Field(default=0.0, ge=0)
    copper_ppm: float = Field(default=0.0, ge=0)
    silver_ppm: float = Field(default=0.0, ge=0)
    sulfur_content: float = Field(default=0.0, ge=0)
    
    # QC parameters
    dilution_factor: float = Field(default=1.0, ge=1)
    measurement_time: float = Field(default=30.0, ge=0)
    calibration_drift: float = Field(default=1.0, ge=0)
    replicate_rsd: float = Field(default=0.0, ge=0)
    blank_intensity: float = Field(default=0.0, ge=0)
    internal_standard: float = Field(default=100.0, ge=0)
    
    # Optional
    client_name: Optional[str] = None
    batch_id: Optional[str] = None
    actual_grade: Optional[float] = None
    notes: Optional[str] = None
    
    @field_validator('replicate_rsd', 'blank_intensity', 'calibration_drift', 'internal_standard', 
                'absorption_242nm', 'absorption_267nm', 'emission_intensity', 'solution_ph', 'temperature_c',
                'ionic_strength', 'dissolved_oxygen', 'iron_ppm', 'copper_ppm', 'silver_ppm', 'sulfur_content',
                'dilution_factor', 'measurement_time', mode='before')
    @classmethod
    def validate_empty_floats(cls, v, info: ValidationInfo):
        if v == '' or v is None:
            # Return appropriate default values based on field name
            field_name = info.field_name
            if field_name == 'solution_ph':
                return 7.0
            elif field_name in ['dilution_factor']:
                return 1.0
            elif field_name in ['measurement_time']:
                return 30.0
            elif field_name in ['calibration_drift', 'internal_standard']:
                return 1.0 if field_name == 'calibration_drift' else 100.0
            else:
                return 0.0
        return v
    
    @field_validator('replicate_rsd')
    @classmethod
    def validate_rsd(cls, v):
        if v > 50:
            raise ValueError('RSD too high (>50%), check measurement quality')
        return v
    
    @field_validator('solution_ph')
    @classmethod
    def validate_ph(cls, v):
        if v is None:
            return 7.0
        if v < 2 or v > 12:
            raise ValueError('pH out of normal range (2-12)')
        return v


class SampleResponse(BaseModel):
    """Response schema for sample analysis."""
    sample_id: str
    predicted_grade: float
    confidence: float
    uncertainty: float
    status: str
    qc_flags: List[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class BatchInput(BaseModel):
    """Input schema for batch processing."""
    batch_name: Optional[str] = None
    client_name: Optional[str] = None
    samples: List[SampleInput]


class BatchResponse(BaseModel):
    """Response schema for batch processing."""
    batch_id: str
    batch_name: Optional[str]
    status: str
    sample_count: int
    completed_count: int
    failed_count: int
    avg_grade: Optional[float]
    min_grade: Optional[float]
    max_grade: Optional[float]
    created_at: datetime
    completed_at: Optional[datetime]


class CalculationRequest(BaseModel):
    """Request schema for calculation tools."""
    type: str = Field(pattern='^(moisture|pulp_density|fineness|weighted_avg|recovery|concentration_ratio)$')
    
    # Moisture calculation
    wet_weight: Optional[float] = None
    dry_weight: Optional[float] = None
    
    # Pulp density
    sg_dry: Optional[float] = None
    sg_pulp: Optional[float] = None
    
    # Fineness
    gold_weight: Optional[float] = None
    total_weight: Optional[float] = None
    
    # Weighted average
    samples: Optional[List[dict]] = None
    
    # Recovery calculation
    feed_grade: Optional[float] = None
    concentrate_grade: Optional[float] = None
    tailings_grade: Optional[float] = None
    
    @field_validator('wet_weight', 'dry_weight')
    @classmethod
    def validate_moisture_weights(cls, v):
        if v is not None and v < 0:
            raise ValueError('Weight cannot be negative')
        return v


class CalculationResponse(BaseModel):
    """Response schema for calculation tools."""
    result: float
    unit: str
    formula_used: str
    status: str


class CalibrationInput(BaseModel):
    """Input schema for calibration record."""
    instrument_id: str
    calibration_type: str = Field(pattern='^(daily|weekly|monthly|maintenance|emergency)$')
    standard_used: str
    drift_check: float
    slope: float
    intercept: float
    r_squared: float
    notes: Optional[str] = None
    
    @field_validator('r_squared')
    @classmethod
    def validate_r_squared(cls, v):
        if v < 0 or v > 1:
            raise ValueError('R² must be between 0 and 1')
        if v < 0.99:
            raise ValueError('R² below acceptable threshold (0.99)')
        return v


class StatisticsFilter(BaseModel):
    """Filter schema for statistical queries."""
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    sample_type: Optional[str] = None
    assay_method: Optional[str] = None
    min_grade: Optional[float] = None
    max_grade: Optional[float] = None
    qc_passed_only: bool = False


class StatisticsResponse(BaseModel):
    """Response schema for statistical analysis."""
    sample_count: int
    avg_grade: float
    median_grade: float
    std_dev: float
    min_grade: float
    max_grade: float
    quartiles: dict
    grade_distribution: List[dict]
    trend_data: List[dict]
    correlation_matrix: Optional[dict]


class UserCreate(BaseModel):
    """Input schema for user creation."""
    username: str = Field(min_length=3, max_length=64)
    email: str
    password: str = Field(min_length=8)
    full_name: Optional[str] = None
    role: str = Field(default='analyst', pattern='^(admin|analyst|technician|viewer)$')


class UserLogin(BaseModel):
    """Input schema for user login."""
    username: str
    password: str


class ExportRequest(BaseModel):
    """Input schema for data export."""
    format: str = Field(default='csv', pattern='^(csv|json|xlsx|pdf)$')
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    sample_type: Optional[str] = None
    include_fields: Optional[List[str]] = None
