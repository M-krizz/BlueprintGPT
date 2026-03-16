#!/usr/bin/env python3
"""
NBC 2018 Rule Audit & Verification System

Performs comprehensive validation of extracted rules:
1. Rule consistency checking
2. Occupancy separation matrix verification
3. Height/area limit cross-validation
4. Fire-resistance rating validation
5. Special requirements completeness checks
6. Edge case and exception handling
"""

import json
import yaml
from typing import Dict, List, Tuple, Set, Any
from dataclasses import dataclass
from enum import Enum


class AuditLevel(Enum):
    """Severity levels for audit findings"""
    CRITICAL = "CRITICAL"      # Rule violation or major data error
    WARNING = "WARNING"         # Potential issue requiring review
    INFO = "INFO"              # Informational note
    PASS = "PASS"              # Rule successfully verified


@dataclass
class AuditFinding:
    """Represents a single audit finding"""
    level: AuditLevel
    category: str              # e.g., "occupancy_separation", "height_limits"
    rule_id: str               # e.g., "RULE_001"
    description: str
    affected_items: List[str]  # What was checked
    recommendation: str        # How to fix if applicable
    nbc_reference: str         # NBC section/table


class RuleAuditor:
    """Audits extracted rules for consistency and completeness"""
    
    def __init__(self, yaml_path: str):
        """Initialize auditor with rules"""
        self.yaml_path = yaml_path
        self.rules = self._load_yaml()
        self.findings: List[AuditFinding] = []
        
    def _load_yaml(self) -> Dict:
        """Load YAML rules"""
        try:
            with open(self.yaml_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Rules file not found: {self.yaml_path}")
    
    def audit_all(self) -> List[AuditFinding]:
        """Run complete audit suite"""
        print("Starting comprehensive rule audit...")
        
        self.findings = []
        
        # Basic structure validation
        self._audit_file_structure()
        print("✓ File structure validation complete")
        
        # Occupancy validation
        self._audit_occupancy_groups()
        print("✓ Occupancy groups validation complete")
        
        # Construction type validation
        self._audit_construction_types()
        print("✓ Construction types validation complete")
        
        # Height/area limits validation
        self._audit_height_area_limits()
        print("✓ Height/area limits validation complete")
        
        # Fire-resistance separations validation
        self._audit_occupancy_separations()
        print("✓ Occupancy separations validation complete")
        
        # Special requirements validation
        self._audit_special_requirements()
        print("✓ Special requirements validation complete")
        
        # Consistency checks
        self._audit_consistency()
        print("✓ Consistency checks complete")
        
        return self.findings
    
    def _audit_file_structure(self) -> None:
        """Verify basic YAML structure"""
        required_sections = [
            'occupancy_groups',
            'construction_types',
            'height_area_limitations',
            'fire_resistance_requirements',
            'means_of_egress'
        ]
        
        for section in required_sections:
            if section not in self.rules:
                self.findings.append(AuditFinding(
                    level=AuditLevel.CRITICAL,
                    category="structure",
                    rule_id="STRUCT_001",
                    description=f"Missing required section: {section}",
                    affected_items=[section],
                    recommendation=f"Add section '{section}' to YAML",
                    nbc_reference="General requirement"
                ))
            else:
                self.findings.append(AuditFinding(
                    level=AuditLevel.PASS,
                    category="structure",
                    rule_id=f"STRUCT_OK_{section}",
                    description=f"Section '{section}' present",
                    affected_items=[section],
                    recommendation="",
                    nbc_reference=""
                ))
    
    def _audit_occupancy_groups(self) -> None:
        """Validate occupancy group definitions"""
        occupancy_groups = self.rules.get('occupancy_groups', {})
        
        # Check all required occupancy groups exist
        required_groups = ['A', 'B', 'E', 'F', 'H', 'I', 'M', 'R', 'S', 'U']
        for group in required_groups:
            if group in occupancy_groups:
                self.findings.append(AuditFinding(
                    level=AuditLevel.PASS,
                    category="occupancy_groups",
                    rule_id=f"OCC_GROUP_{group}",
                    description=f"Occupancy group {group} defined",
                    affected_items=[group],
                    recommendation="",
                    nbc_reference="Chapter 3"
                ))
            else:
                self.findings.append(AuditFinding(
                    level=AuditLevel.CRITICAL,
                    category="occupancy_groups",
                    rule_id=f"OCC_GROUP_MISSING_{group}",
                    description=f"Occupancy group {group} not defined",
                    affected_items=[group],
                    recommendation=f"Add occupancy group {group}",
                    nbc_reference="Chapter 3, Table 302.1"
                ))
        
        # Validate H-occupancy subgroups (H-1 through H-5)
        h_groups = occupancy_groups.get('H', {})
        h_subgroups = h_groups.get('subgroups', {})
        required_h = ['H-1', 'H-2', 'H-3', 'H-4', 'H-5']
        
        for h in required_h:
            h_code = h.split('-')[1]
            if h_code in h_subgroups:
                self.findings.append(AuditFinding(
                    level=AuditLevel.PASS,
                    category="occupancy_subgroups",
                    rule_id=f"H_SUB_{h}",
                    description=f"High-hazard subgroup {h} defined",
                    affected_items=[h],
                    recommendation="",
                    nbc_reference="Chapter 4, Section 415"
                ))
    
    def _audit_construction_types(self) -> None:
        """Validate construction type definitions"""
        construction_types = self.rules.get('construction_types', {})
        
        required_types = ['I-A', 'I-B', 'II-A', 'II-B', 'II-C', 'III-A', 'III-B', 'IV-A', 'V-A', 'V-B']
        
        for ctype in required_types:
            if ctype in construction_types:
                self.findings.append(AuditFinding(
                    level=AuditLevel.PASS,
                    category="construction_types",
                    rule_id=f"TYPE_{ctype}",
                    description=f"Construction type {ctype} defined",
                    affected_items=[ctype],
                    recommendation="",
                    nbc_reference="Chapter 6"
                ))
            else:
                self.findings.append(AuditFinding(
                    level=AuditLevel.CRITICAL,
                    category="construction_types",
                    rule_id=f"TYPE_MISSING_{ctype}",
                    description=f"Construction type {ctype} not defined",
                    affected_items=[ctype],
                    recommendation=f"Add construction type {ctype}",
                    nbc_reference="Chapter 6, Table 601"
                ))
        
        # Validate fire ratings are ordered correctly
        for ctype, data in construction_types.items():
            frame_rating = data.get('fire_rating_structural_frame', 0)
            wall_rating = data.get('fire_rating_exterior_walls', 0)
            bearing_rating = data.get('fire_rating_interior_bearing_walls', 0)
            
            # Structural frame should not exceed wall or bearing ratings
            if frame_rating > wall_rating or frame_rating > bearing_rating:
                self.findings.append(AuditFinding(
                    level=AuditLevel.WARNING,
                    category="construction_types",
                    rule_id=f"TYPE_RATING_{ctype}",
                    description=f"Type {ctype}: Frame rating ({frame_rating}) exceeds wall/bearing ratings",
                    affected_items=[ctype],
                    recommendation="Review fire rating hierarchy",
                    nbc_reference="Chapter 6"
                ))
    
    def _audit_height_area_limits(self) -> None:
        """Validate height and area limit tables"""
        limits = self.rules.get('height_area_limitations', {})
        
        height_limits = limits.get('height_limits_feet', {})
        if not height_limits:
            self.findings.append(AuditFinding(
                level=AuditLevel.CRITICAL,
                category="height_limits",
                rule_id="HEIGHT_MISSING",
                description="Height limits table (Table 504.3) not found",
                affected_items=["height_limits_feet"],
                recommendation="Extract Table 504.3 from NBC Chapter 5",
                nbc_reference="Chapter 5, Table 504.3"
            ))
        else:
            self.findings.append(AuditFinding(
                level=AuditLevel.PASS,
                category="height_limits",
                rule_id="HEIGHT_FOUND",
                description=f"Height limits table found with {len(height_limits)} entries",
                affected_items=list(height_limits.keys()),
                recommendation="",
                nbc_reference="Chapter 5, Table 504.3"
            ))
            
            # Validate that all construction types have height limits
            required_types = ['I-A', 'I-B', 'II-A', 'II-B', 'II-C', 'III-A', 'III-B', 'IV-A', 'V-A', 'V-B']
            missing_types = set(required_types) - set(height_limits.keys())
            if missing_types:
                self.findings.append(AuditFinding(
                    level=AuditLevel.WARNING,
                    category="height_limits",
                    rule_id="HEIGHT_INCOMPLETE",
                    description=f"Height limits missing for types: {missing_types}",
                    affected_items=list(missing_types),
                    recommendation="Add height limits for all construction types",
                    nbc_reference="Chapter 5, Table 504.3"
                ))
        
        area_limits = limits.get('area_limits_sqft', {})
        if not area_limits:
            self.findings.append(AuditFinding(
                level=AuditLevel.CRITICAL,
                category="area_limits",
                rule_id="AREA_MISSING",
                description="Area limits table (Table 506.2) not found",
                affected_items=["area_limits_sqft"],
                recommendation="Extract Table 506.2 from NBC Chapter 5",
                nbc_reference="Chapter 5, Table 506.2"
            ))
        else:
            self.findings.append(AuditFinding(
                level=AuditLevel.PASS,
                category="area_limits",
                rule_id="AREA_FOUND",
                description=f"Area limits table found with {len(area_limits)} entries",
                affected_items=list(area_limits.keys()),
                recommendation="",
                nbc_reference="Chapter 5, Table 506.2"
            ))
    
    def _audit_occupancy_separations(self) -> None:
        """Validate occupancy separation fire ratings"""
        separations = self.rules.get('fire_resistance_requirements', {}).get('occupancy_separations', {})
        
        if not separations:
            self.findings.append(AuditFinding(
                level=AuditLevel.CRITICAL,
                category="separations",
                rule_id="SEP_MISSING",
                description="Occupancy separation matrix (Table 508.4) not found",
                affected_items=["occupancy_separations"],
                recommendation="Extract Table 508.4 from NBC Chapter 5",
                nbc_reference="Chapter 5, Table 508.4"
            ))
        else:
            self.findings.append(AuditFinding(
                level=AuditLevel.PASS,
                category="separations",
                rule_id="SEP_FOUND",
                description=f"Occupancy separation matrix found with {len(separations)} entries",
                affected_items=list(separations.keys())[:5],  # Show first 5
                recommendation="",
                nbc_reference="Chapter 5, Table 508.4"
            ))
            
            # Validate ratings are 0, 1, 2, 3, or 4 hours
            valid_ratings = {0, 1, 2, 3, 4}
            invalid_ratings = {}
            
            for key, rating in separations.items():
                if rating not in valid_ratings:
                    if key not in invalid_ratings:
                        invalid_ratings[key] = rating
            
            if invalid_ratings:
                self.findings.append(AuditFinding(
                    level=AuditLevel.CRITICAL,
                    category="separations",
                    rule_id="SEP_INVALID_RATING",
                    description=f"Invalid fire ratings found in separations: {invalid_ratings}",
                    affected_items=list(invalid_ratings.keys()),
                    recommendation="Fire ratings must be 0, 1, 2, 3, or 4 hours",
                    nbc_reference="Chapter 5, Table 508.4"
                ))
    
    def _audit_special_requirements(self) -> None:
        """Validate special requirements are present"""
        required_sections = {
            'means_of_egress': 'Means of Egress Requirements',
            'interior_finish': 'Interior Finish Classifications',
            'lighting_ventilation': 'Lighting and Ventilation'
        }
        
        for section, description in required_sections.items():
            if section in self.rules:
                self.findings.append(AuditFinding(
                    level=AuditLevel.PASS,
                    category="special_requirements",
                    rule_id=f"SPEC_{section}",
                    description=f"{description} defined",
                    affected_items=[section],
                    recommendation="",
                    nbc_reference="Chapter 4"
                ))
            else:
                self.findings.append(AuditFinding(
                    level=AuditLevel.WARNING,
                    category="special_requirements",
                    rule_id=f"SPEC_{section}_MISSING",
                    description=f"{description} not found",
                    affected_items=[section],
                    recommendation=f"Consider adding {description} section",
                    nbc_reference="Chapter 4"
                ))
    
    def _audit_consistency(self) -> None:
        """Check cross-section consistency"""
        occupancy_groups = set(self.rules.get('occupancy_groups', {}).keys())
        height_limits = self.rules.get('height_area_limitations', {}).get('height_limits_feet', {})
        
        # Check if height limits reference all occupancy groups
        for const_type, occupancies in height_limits.items():
            defined_occupancies = set(occupancies.keys()) if occupancies else set()
            
            # For Type I-A, should have many occupancies
            if const_type == 'I-A':
                coverage_ratio = len(defined_occupancies) / len(occupancy_groups)
                if coverage_ratio < 0.8:
                    self.findings.append(AuditFinding(
                        level=AuditLevel.WARNING,
                        category="consistency",
                        rule_id=f"CONSIST_HEIGHT_{const_type}",
                        description=f"Type {const_type} height limits only cover {coverage_ratio*100:.0f}% of occupancies",
                        affected_items=[const_type],
                        recommendation="Verify height limits are complete for all occupancies",
                        nbc_reference="Chapter 5, Table 504.3"
                    ))


class AuditReporter:
    """Generates audit reports"""
    
    @staticmethod
    def report_text(findings: List[AuditFinding]) -> str:
        """Generate text audit report"""
        lines = [
            "=" * 80,
            "NBC 2018 RULE AUDIT REPORT",
            "=" * 80,
        ]
        
        # Summary statistics
        levels = {}
        for f in findings:
            level = f.level.value
            levels[level] = levels.get(level, 0) + 1
        
        lines.append(f"\nSummary:")
        lines.append(f"  Total Findings:  {len(findings)}")
        lines.append(f"  CRITICAL:        {levels.get('CRITICAL', 0)}")
        lines.append(f"  WARNING:         {levels.get('WARNING', 0)}")
        lines.append(f"  PASS:            {levels.get('PASS', 0)}")
        lines.append(f"  INFO:            {levels.get('INFO', 0)}")
        
        # Detailed findings by level
        for level in [AuditLevel.CRITICAL, AuditLevel.WARNING, AuditLevel.INFO]:
            level_findings = [f for f in findings if f.level == level]
            if level_findings:
                lines.append(f"\n{'-'*80}")
                lines.append(f"{level.value} FINDINGS ({len(level_findings)}):")
                lines.append(f"{'-'*80}")
                
                for i, f in enumerate(level_findings[:10], 1):  # Show first 10
                    lines.append(f"\n{i}. [{f.rule_id}] {f.category}")
                    lines.append(f"   {f.description}")
                    if f.affected_items:
                        lines.append(f"   Affected: {', '.join(f.affected_items[:3])}")
                    if f.recommendation:
                        lines.append(f"   Recommendation: {f.recommendation}")
                    lines.append(f"   Reference: {f.nbc_reference}")
        
        lines.append("\n" + "=" * 80)
        
        # Overall assessment
        critical_count = levels.get('CRITICAL', 0)
        warning_count = levels.get('WARNING', 0)
        
        if critical_count > 0:
            lines.append("STATUS: AUDIT FAILED - Critical issues found")
        elif warning_count > 5:
            lines.append("STATUS: AUDIT PASSED WITH WARNINGS - Review recommendations")
        else:
            lines.append("STATUS: AUDIT PASSED - Rules verified")
        
        lines.append("=" * 80)
        return "\n".join(lines)
    
    @staticmethod
    def report_json(findings: List[AuditFinding]) -> Dict:
        """Generate JSON audit report"""
        return {
            'summary': {
                'total_findings': len(findings),
                'critical': sum(1 for f in findings if f.level == AuditLevel.CRITICAL),
                'warnings': sum(1 for f in findings if f.level == AuditLevel.WARNING),
                'pass': sum(1 for f in findings if f.level == AuditLevel.PASS),
            },
            'findings': [
                {
                    'level': f.level.value,
                    'category': f.category,
                    'rule_id': f.rule_id,
                    'description': f.description,
                    'affected_items': f.affected_items,
                    'recommendation': f.recommendation,
                    'nbc_reference': f.nbc_reference
                }
                for f in findings
            ]
        }


def main():
    """Main audit entry point"""
    import sys
    
    yaml_path = "ground_truth/ground_truth.yml"
    
    try:
        auditor = RuleAuditor(yaml_path)
        findings = auditor.audit_all()
        
        # Generate reports
        print("\n" + AuditReporter.report_text(findings))
        
        # Export JSON report
        report = AuditReporter.report_json(findings)
        with open("ground_truth/audit_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Audit report exported to ground_truth/audit_report.json")
        
        # Determine exit code
        critical_count = report['summary']['critical']
        if critical_count > 0:
            print(f"\n⚠ {critical_count} CRITICAL issues require attention")
            return 1
        else:
            print(f"\n✓ Audit completed successfully")
            return 0
    
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
