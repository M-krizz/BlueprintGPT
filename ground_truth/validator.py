#!/usr/bin/env python3
"""
NBC Compliance Scorer & Validator

Provides:
1. Scoring system for compliance assessment
2. Rule validation engine
3. Building configuration compliance checking
4. Violation reporting
"""

import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class BuildingConfiguration:
    """Represents a building design configuration"""
    occupancy_group: str                    # e.g., "A-1", "B", "R-2"
    construction_type: str                  # e.g., "I-A", "II-B", "V-A"
    height_feet: float
    area_sqft: float
    num_stories: int
    has_sprinklers: bool = False
    has_fire_alarm: bool = False
    fire_separation_hours: int = 0
    mixed_occupancies: List[str] = field(default_factory=list)  # For mixed-use buildings


@dataclass
class ComplianceViolation:
    """Represents a compliance violation"""
    rule_id: str
    rule_description: str
    severity: str                          # "CRITICAL", "MAJOR", "MINOR"
    building_value: Any
    required_value: Any
    remediation: str
    rule_section: str


@dataclass
class ComplianceScore:
    """Score result for a building configuration"""
    configuration: BuildingConfiguration
    score: float                           # 0.0 to 100.0
    violations: List[ComplianceViolation]
    status: str                            # "COMPLIANT", "NON_COMPLIANT", "CONDITIONAL"
    details: Dict[str, Any] = field(default_factory=dict)


class ComplianceValidator:
    """Validates building configurations against NBC rules"""
    
    def __init__(self, rules_file: str, constraints_file: str):
        """Initialize validator with rules"""
        self.rules = self._load_json(rules_file)
        self.constraints = self._load_json(constraints_file)
        
        self.violations: List[ComplianceViolation] = []
    
    def _load_json(self, path: str) -> Dict:
        """Load JSON file"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Rules file not found: {path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
    
    def validate_configuration(self, config: BuildingConfiguration) -> ComplianceScore:
        """
        Validate a building configuration against all rules
        Returns compliance score and list of violations
        """
        self.violations = []
        
        # Check height limits
        self._check_height_limits(config)
        
        # Check area limits
        self._check_area_limits(config)
        
        # Check construction type restrictions
        self._check_construction_type_allowed(config)
        
        # Check fire-resistance requirements
        self._check_fire_resistance(config)
        
        # Check mixed occupancy rules
        if config.mixed_occupancies:
            self._check_mixed_occupancy(config)
        
        # Calculate score
        score = self._calculate_score(config)
        
        # Determine status
        if not self.violations:
            status = "COMPLIANT"
        elif all(v.severity == "MINOR" for v in self.violations):
            status = "CONDITIONAL"
        else:
            status = "NON_COMPLIANT"
        
        return ComplianceScore(
            configuration=config,
            score=score,
            violations=self.violations.copy(),
            status=status,
            details=self._build_details(config)
        )
    
    def _check_height_limits(self, config: BuildingConfiguration) -> None:
        """Check building height against occupancy/type limits"""
        # This is a simplified check - in reality would lookup tables
        # For now, create sample rules
        
        # Example: Type V-B has 65 ft limit for most occupancies
        if config.construction_type == "V-B":
            max_height = 65.0
            if config.height_feet > max_height:
                violation = ComplianceViolation(
                    rule_id="height_limit_V-B",
                    rule_description="Type V-B construction height limit",
                    severity="CRITICAL",
                    building_value=config.height_feet,
                    required_value=max_height,
                    remediation="Reduce building height or upgrade construction type",
                    rule_section="Table 504.3"
                )
                self.violations.append(violation)
    
    def _check_area_limits(self, config: BuildingConfiguration) -> None:
        """Check building area against occupancy/type limits"""
        # Example limits by occupancy and construction type
        area_limits = {
            ("B", "V-B"): 9_000,           # 9,000 sq ft
            ("B", "II-C"): 65_000,         # 65,000 sq ft
            ("A-3", "V-B"): 6_000,         # 6,000 sq ft
            ("A-3", "I-A"): None,          # Unlimited
        }
        
        key = (config.occupancy_group, config.construction_type)
        if key in area_limits:
            limit = area_limits[key]
            if limit and config.area_sqft > limit:
                # Can be mitigated with sprinklers (25% increase)
                sprinkler_limit = limit * 1.25 if config.has_sprinklers else limit
                
                if config.area_sqft > sprinkler_limit:
                    severity = "CRITICAL" if config.area_sqft > sprinkler_limit * 1.1 else "MAJOR"
                    violation = ComplianceViolation(
                        rule_id="area_limit",
                        rule_description=f"Area limit for {config.occupancy_group} in {config.construction_type}",
                        severity=severity,
                        building_value=config.area_sqft,
                        required_value=sprinkler_limit,
                        remediation="Reduce building area or install fire suppression system",
                        rule_section="Table 506.2 / Section 903"
                    )
                    self.violations.append(violation)
    
    def _check_construction_type_allowed(self, config: BuildingConfiguration) -> None:
        """Check if construction type is allowed for occupancy"""
        # H occupancies have severe restrictions
        if config.occupancy_group.startswith("H-"):
            allowed_types = ["I-A", "I-B", "II-A"]
            if config.construction_type not in allowed_types:
                violation = ComplianceViolation(
                    rule_id="h_occupancy_type_restriction",
                    rule_description=f"High-hazard occupancy {config.occupancy_group} not allowed in {config.construction_type}",
                    severity="CRITICAL",
                    building_value=config.construction_type,
                    required_value=allowed_types,
                    remediation="Use Type I-A, I-B, or II-A construction",
                    rule_section="Table 503.1 / Section 415"
                )
                self.violations.append(violation)
        
        # R-3 (one/two family) can use Type V
        if config.occupancy_group == "R-3":
            if config.construction_type not in ["V-A", "V-B"]:
                violation = ComplianceViolation(
                    rule_id="r3_type_warning",
                    rule_description=f"Type {config.construction_type} unusual for R-3 occupancy",
                    severity="MINOR",
                    building_value=config.construction_type,
                    required_value=["V-A", "V-B"],
                    remediation="Consider Type V construction for cost efficiency",
                    rule_section="Table 503.1"
                )
                self.violations.append(violation)
    
    def _check_fire_resistance(self, config: BuildingConfiguration) -> None:
        """Check fire-resistance ratings for mixed occupancy"""
        if config.mixed_occupancies:
            # If H-type occupancy in mixed building, high ratings required
            if any(occ.startswith("H-") for occ in config.mixed_occupancies):
                if config.fire_separation_hours < 4:
                    violation = ComplianceViolation(
                        rule_id="high_hazard_separation",
                        rule_description="High-hazard occupancy requires 4-hour fire separation",
                        severity="CRITICAL",
                        building_value=config.fire_separation_hours,
                        required_value=4,
                        remediation="Provide 4-hour fire-resistance-rated barriers",
                        rule_section="Table 508.4"
                    )
                    self.violations.append(violation)
            
            # General separation checks
            if config.fire_separation_hours < 2 and len(config.mixed_occupancies) > 1:
                violation = ComplianceViolation(
                    rule_id="mixed_occupancy_separation",
                    rule_description="Mixed occupancy requires minimum 2-hour separation",
                    severity="MAJOR",
                    building_value=config.fire_separation_hours,
                    required_value=2,
                    remediation="Increase fire-resistance rating to minimum 2 hours",
                    rule_section="Section 508"
                )
                self.violations.append(violation)
    
    def _check_mixed_occupancy(self, config: BuildingConfiguration) -> None:
        """Check mixed-occupancy specific rules"""
        if len(config.mixed_occupancies) < 2:
            return
        
        # Nonseparated occupancy rules (default assumption)
        # Most restrictive requirements apply
        most_restrictive = self._get_most_restrictive_occupancy(config.mixed_occupancies)
        
        # Warn if using less restrictive main occupancy
        if config.occupancy_group != most_restrictive:
            violation = ComplianceViolation(
                rule_id="mixed_occ_classification",
                rule_description="Classification should follow most restrictive occupancy",
                severity="MINOR",
                building_value=config.occupancy_group,
                required_value=most_restrictive,
                remediation="Classify building as the most restrictive occupancy present",
                rule_section="Section 508.3"
            )
            self.violations.append(violation)
    
    def _get_most_restrictive_occupancy(self, occupancies: List[str]) -> str:
        """Determine most restrictive occupancy from list"""
        # H > I > R > A > F > E > M > B > S > U (example hierarchy)
        hierarchy = {"H": 10, "I": 9, "R": 8, "A": 7, "F": 6, "E": 5, "M": 4, "B": 3, "S": 2, "U": 1}
        return max(occupancies, key=lambda x: hierarchy.get(x[0], 0))
    
    def _calculate_score(self, config: BuildingConfiguration) -> float:
        """Calculate compliance score from violations"""
        if not self.violations:
            return 100.0
        
        # Scoring: critical = -20 points, major = -10, minor = -5
        score = 100.0
        for violation in self.violations:
            if violation.severity == "CRITICAL":
                score -= 20
            elif violation.severity == "MAJOR":
                score -= 10
            elif violation.severity == "MINOR":
                score -= 5
        
        return max(0.0, score)
    
    def _build_details(self, config: BuildingConfiguration) -> Dict[str, Any]:
        """Build detailed compliance information"""
        return {
            'occupancy': config.occupancy_group,
            'construction_type': config.construction_type,
            'height': config.height_feet,
            'area': config.area_sqft,
            'stories': config.num_stories,
            'fire_systems': {
                'sprinklers': config.has_sprinklers,
                'fire_alarm': config.has_fire_alarm,
                'separation_hours': config.fire_separation_hours
            },
            'violation_summary': {
                'total': len(self.violations),
                'critical': sum(1 for v in self.violations if v.severity == "CRITICAL"),
                'major': sum(1 for v in self.violations if v.severity == "MAJOR"),
                'minor': sum(1 for v in self.violations if v.severity == "MINOR")
            }
        }


class ComplianceReporter:
    """Generates compliance reports"""
    
    @staticmethod
    def report_text(score: ComplianceScore) -> str:
        """Generate text report"""
        config = score.configuration
        lines = [
            "=" * 70,
            "NBC COMPLIANCE ASSESSMENT REPORT",
            "=" * 70,
            f"\nBuilding Configuration:",
            f"  Occupancy Group:     {config.occupancy_group}",
            f"  Construction Type:   {config.construction_type}",
            f"  Height:              {config.height_feet} ft",
            f"  Area:                {config.area_sqft} sq ft",
            f"  Stories:             {config.num_stories}",
            f"  Has Sprinklers:      {config.has_sprinklers}",
            f"\nCompliance Status:    {score.status}",
            f"Compliance Score:     {score.score:.1f}/100.0",
        ]
        
        if score.violations:
            lines.append(f"\nViolations Found: {len(score.violations)}")
            lines.append("-" * 70)
            for v in score.violations:
                lines.append(f"\n  [{v.severity}] {v.rule_id}")
                lines.append(f"    Rule: {v.rule_description}")
                lines.append(f"    Building Value: {v.building_value}")
                lines.append(f"    Required Value: {v.required_value}")
                lines.append(f"    Remediation: {v.remediation}")
                lines.append(f"    Reference: {v.rule_section}")
        else:
            lines.append("\n✓ No violations found - Building is fully compliant!")
        
        lines.append("\n" + "=" * 70)
        return "\n".join(lines)
    
    @staticmethod
    def report_json(score: ComplianceScore) -> Dict:
        """Generate JSON report"""
        return {
            'configuration': {
                'occupancy': score.configuration.occupancy_group,
                'construction_type': score.configuration.construction_type,
                'height_feet': score.configuration.height_feet,
                'area_sqft': score.configuration.area_sqft,
                'stories': score.configuration.num_stories
            },
            'compliance': {
                'status': score.status,
                'score': score.score,
                'violations': [
                    {
                        'rule_id': v.rule_id,
                        'severity': v.severity,
                        'description': v.rule_description,
                        'building_value': v.building_value,
                        'required_value': v.required_value,
                        'remediation': v.remediation,
                        'rule_section': v.rule_section
                    }
                    for v in score.violations
                ]
            },
            'details': score.details
        }


def example_validation():
    """Example usage of validator"""
    # Create example configurations
    configs = [
        BuildingConfiguration(
            occupancy_group="B",
            construction_type="II-B",
            height_feet=75,
            area_sqft=50_000,
            num_stories=5,
            has_sprinklers=True
        ),
        BuildingConfiguration(
            occupancy_group="A-3",
            construction_type="V-B",
            height_feet=100,
            area_sqft=25_000,
            num_stories=8,
            has_sprinklers=False
        ),
        BuildingConfiguration(
            occupancy_group="H-2",
            construction_type="V-B",
            height_feet=55,
            area_sqft=15_000,
            num_stories=2,
            has_sprinklers=True
        ),
    ]
    
    # Validate each
    try:
        validator = ComplianceValidator(
            "ground_truth/ground_truth.yml",
            "ground_truth/constraints.json"
        )
        
        for config in configs:
            score = validator.validate_configuration(config)
            print(ComplianceReporter.report_text(score))
            print()
    except Exception as e:
        print(f"Validator initialization note: {e}")
        print("(This is expected if compliance files aren't ready yet)")


if __name__ == "__main__":
    example_validation()
