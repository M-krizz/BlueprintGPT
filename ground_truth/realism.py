#!/usr/bin/env python3
"""
NBC Compliance Solver - Realism Upgrade System

Adds real-world design constraints and considerations:
1. Practical design constraints (typical vs. edge cases)
2. Cost/feasibility factors
3. Owner requirements and preferences
4. Regulatory variance common practices
5. Industry standard practices
6. Construction feasibility metrics
"""

import json
from typing import Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class RealismUpgrade:
    """Represents a realism upgrade to the compliance system"""
    upgrade_id: str
    name: str
    category: str              # e.g., "cost_factors", "design_preferences"
    description: str
    impact: str                # How it affects compliance checking
    applies_to: List[str]      # Occupancy groups or design types


class RealismUpgrader:
    """Adds real-world design considerations"""
    
    def __init__(self):
        """Initialize upgrader"""
        self.upgrades: Dict[str, Any] = {}
        
    def generate_upgrades(self) -> Dict[str, Any]:
        """Generate all realism upgrades"""
        print("Generating realism upgrades...")
        
        self.upgrades = {
            'cost_efficiency_factors': self._create_cost_factors(),
            'design_preferences': self._create_design_preferences(),
            'construction_feasibility': self._create_construction_feasibility(),
            'regulatory_practices': self._create_regulatory_practices(),
            'occupancy_specific': self._create_occupancy_specific(),
            'fire_systems': self._create_fire_systems(),
        }
        
        return self.upgrades
    
    def _create_cost_factors(self) -> Dict:
        """Create cost efficiency factors for designs"""
        return {
            'construction_type_costs': {
                'description': 'Relative cost multiplier for construction types (normalized to Type V-B = 1.0)',
                'factors': {
                    'V-B': 1.0,         # Baseline
                    'V-A': 1.15,        # Protected combustible
                    'IV-A': 1.25,       # Heavy timber (specialty labor)
                    'III-B': 1.30,      # Unprotected combustible
                    'III-A': 1.45,      # Protected combustible (masonry)
                    'II-C': 1.50,       # Unprotected non-combustible
                    'II-B': 1.65,       # Protected non-combustible
                    'II-A': 1.80,       # Non-combustible with ratings
                    'I-B': 2.10,        # Modified fire-resistant
                    'I-A': 2.50,        # Full fire-resistant (premium)
                },
                'notes': [
                    'Costs can vary significantly by region',
                    'Special materials (heavy timber, fire-rated) command premiums',
                    'Prefab vs. site-built affects costs',
                    'Labor availability affects costs'
                ]
            },
            'fire_suppression_costs': {
                'sprinkler_system': {
                    'cost_per_sqft': 1.50,    # Typical $1-2 per sq ft
                    'installation_time_days': 20,
                    'benefits': ['25% area increase', 'Height increase (some cases)', 'Lower insurance']
                },
                'fire_alarm_system': {
                    'cost_per_sqft': 0.50,    # Typical $0.30-0.70
                    'installation_time_days': 10,
                    'benefits': ['Enhanced egress', 'Occupancy increase potential']
                },
                'smoke_control': {
                    'cost_per_sqft': 2.00,    # Typical for high-rise
                    'installation_time_days': 30,
                    'benefits': ['High-rise building permit', 'Safety enhancement']
                }
            },
            'typical_budgets': {
                'description': 'Typical construction budgets by occupancy ($/sq ft)',
                'A-1': {'low': 250, 'mid': 350, 'high': 500},
                'B': {'low': 150, 'mid': 200, 'high': 300},
                'E': {'low': 180, 'mid': 250, 'high': 350},
                'F-1': {'low': 120, 'mid': 150, 'high': 200},
                'M': {'low': 180, 'mid': 250, 'high': 350},
                'R-2': {'low': 180, 'mid': 250, 'high': 400},
                'S-1': {'low': 100, 'mid': 130, 'high': 180},
            }
        }
    
    def _create_design_preferences(self) -> Dict:
        """Create common design preferences and constraints"""
        return {
            'owner_requirements': {
                'description': 'Common owner/developer requirements that may override pure code compliance',
                'examples': [
                    {
                        'name': 'Minimum story count',
                        'description': 'Developer wants 5+ stories for parking/retail efficiency',
                        'impact': 'Forces higher construction type (e.g., II-B instead of V-B)',
                        'mitigation': 'May justify cost of better construction type'
                    },
                    {
                        'name': 'Minimum floor size',
                        'description': 'Tenant wants 100K+ sq ft per story for large office',
                        'impact': 'May exceed type limits, requires sprinklers or area separation',
                        'mitigation': 'Sprinkler system becomes mandatory'
                    },
                    {
                        'name': 'Mixed-use vs. pure occupancy',
                        'description': 'Developer wants ground floor retail + upper office',
                        'impact': 'Mixed-occupancy rules apply (more restrictive)',
                        'mitigation': 'Consider fire separation vs. nonseparated approach'
                    }
                ]
            },
            'site_constraints': {
                'description': 'Real-world constraints that limit design options',
                'examples': [
                    {
                        'name': 'Lot size',
                        'description': 'Small urban lot limits building footprint',
                        'impact': 'Must build taller to meet area requirements',
                        'mitigation': 'Requires Type I/II or sprinklers'
                    },
                    {
                        'name': 'Height restrictions (zoning)',
                        'description': 'Municipal zoning limits height to 85 feet',
                        'impact': 'May conflict with code requirements',
                        'mitigation': 'Seek variance or modify occupancy mix'
                    },
                    {
                        'name': 'Historic building (renovation)',
                        'description': 'Strict preservation rules (can\'t alter exterior)',
                        'impact': 'May preclude certain construction types',
                        'mitigation': 'Seek historic equivalency variances'
                    }
                ]
            }
        }
    
    def _create_construction_feasibility(self) -> Dict:
        """Create construction feasibility metrics"""
        return {
            'construction_type_availability': {
                'description': 'Typical contractor expertise and material availability by region',
                'regions': {
                    'urban_dense': {
                        'preferred_types': ['I-A', 'I-B', 'II-A', 'II-B'],
                        'reason': 'High-rise, complex projects common',
                        'limitations': 'Type V contractors less common',
                        'expertise': 'Fire-resistant, precast common'
                    },
                    'suburban': {
                        'preferred_types': ['II-B', 'II-C', 'III-A', 'V-A'],
                        'reason': 'Mid-rise mixed buildings',
                        'limitations': 'Type I-A rarely needed',
                        'expertise': 'Wood/steel hybrid construction'
                    },
                    'rural': {
                        'preferred_types': ['V-A', 'V-B', 'IV-A'],
                        'reason': 'Lower-rise, simpler buildings',
                        'limitations': 'Type I-A specialty contractors needed',
                        'expertise': 'Wood frame and basic construction'
                    }
                }
            },
            'material_lead_times': {
                'description': 'Typical lead time impact on project schedule',
                'standard': 'On-shelf materials (weeks)',
                'specialty': {
                    'fire_rated_assemblies': '6-12 weeks',
                    'precast_concrete': '8-12 weeks',
                    'structural_steel': '4-8 weeks',
                    'curtain_walls': '6-10 weeks'
                }
            },
            'labor_complexity': {
                'description': 'Labor skill requirements by construction type',
                'V-B': {'level': 'Entry', 'hours_per_sqft': 1.5},
                'V-A': {'level': 'Intermediate', 'hours_per_sqft': 1.8},
                'III-B': {'level': 'Intermediate', 'hours_per_sqft': 1.7},
                'III-A': {'level': 'Advanced', 'hours_per_sqft': 2.0},
                'II-C': {'level': 'Advanced', 'hours_per_sqft': 2.2},
                'II-A': {'level': 'Expert', 'hours_per_sqft': 2.8},
                'I-A': {'level': 'Expert', 'hours_per_sqft': 3.5},
            }
        }
    
    def _create_regulatory_practices(self) -> Dict:
        """Create common regulatory variance practices"""
        return {
            'common_variances': {
                'description': 'Commonly approved regulatory variances/equivalencies',
                'examples': [
                    {
                        'name': 'Historic Building Equivalency',
                        'applies_to': 'Buildings 50+ years old',
                        'typical_approach': 'Demonstrate alternative protection (e.g., sprinklers)',
                        'approval_rate': '85%+',
                        'timeline_weeks': 6
                    },
                    {
                        'name': 'Fire-Resistance Equivalency',
                        'applies_to': 'When fire-rated assemblies unavailable',
                        'typical_approach': 'Use prescriptive assembly or fire testing',
                        'approval_rate': '75%+',
                        'timeline_weeks': 12
                    },
                    {
                        'name': 'Area Modification Credit',
                        'applies_to': 'Beyond sprinkler credits',
                        'typical_approach': 'Frontage on public way (25%+), open perimeter',
                        'approval_rate': '90%+',
                        'timeline_weeks': 2
                    }
                ]
            },
            'performance_path_alternatives': {
                'description': 'Alternatives to prescriptive code compliance',
                'fire_safety_engineering': {
                    'approach': 'Fire safety analysis instead of prescriptive rules',
                    'requirements': ['Professional engineer', 'Fire modeling', 'Tenant approval'],
                    'timeline_weeks': '8-16',
                    'cost_premium': '10-15%'
                },
                'constructed_performance': {
                    'approach': 'Demonstrate equivalency through construction/testing',
                    'examples': ['Full-scale fire tests', 'Approved precedents'],
                    'timeline_weeks': '12-24',
                    'cost_premium': '20-30%'
                }
            }
        }
    
    def _create_occupancy_specific(self) -> Dict:
        """Create occupancy-specific realism upgrades"""
        return {
            'A_assembly': {
                'common_issues': [
                    'Exit capacity severely constraints occupancy (250 ft limit)',
                    'Multiple exits required (large buildings need 3+)',
                    'Capacity often limited by parking availability'
                ],
                'typical_workarounds': [
                    'Add multiple stairwells (large vertical expense)',
                    'Reduce occupant load vs. floor area (tenant complaint)',
                    'Valet parking instead of attendant parking'
                ],
                'realistic_size_limits': {
                    'single_floor': '5000-10000 sq ft',
                    'multi_floor': '15000-25000 sq ft total',
                    'note': 'Practical limits; code allows larger'
                }
            },
            'B_business': {
                'common_issues': [
                    'Office tenants want 300 sq ft per person (code allows 250)',
                    'Open floor plates desired (60000+ sq ft)',
                    'Multiple egress requirement constrains layout'
                ],
                'typical_solutions': [
                    'Sprinkler system ($1.50/sq ft typical)',
                    'Central core with atrium (often desired anyway)',
                    'Type II construction for economics'
                ],
                'realistic_footprint': '50000-80000 sq ft per floor'
            },
            'E_educational': {
                'common_issues': [
                    '250 ft exit travel limit per classroom', 
                    'Owner wants "safe routes" to parking (beyond code)',
                    'Multiple stairwell requirements increase cost'
                ],
                'typical_solutions': [
                    'Classroom wings with stairwells every 100-120 ft',
                    'Central commons as safety rally point',
                    'Type III minimum for cost'
                ],
                'standard_sizes': {
                    'elementary': '60000-80000 sq ft',
                    'high_school': '200000-300000 sq ft',
                    'university': '100000+ sq ft per building'
                }
            },
            'H_high_hazard': {
                'common_issues': [
                    'Extremely expensive to build (Type I-A only)',
                    'Industry standards often exceed code',
                    'Insurance costs prohibitive without sprinklers'
                ],
                'typical_solutions': [
                    'H-2, H-3 more common than H-1 in practice',
                    'Full sprinklers usually mandatory',
                    'Frequent regulatory consultation'
                ],
                'realistic_size': '5000-20000 sq ft (very restrictive)'
            },
            'R_residential': {
                'common_issues': [
                    'R-2 (multifamily) economics drive much development',
                    'Sprinklers often added by code (some states)',
                    'Owner wants ground floor retail (mixed-use complexity)'
                ],
                'typical_solutions': [
                    'Type V-A with sprinklers (most economical)',
                    'Separate retail with fire walls (Table 508.4)',
                    'Mixed-use Type II-B becomes necessary'
                ],
                'market_drivers': {
                    'urban_large': 'Type II-B for 8-12 stories',
                    'suburban_mid': 'Type V-A sprinklered, 4-6 stories',
                    'rural_small': 'Type V-B, 3-4 stories'
                }
            }
        }
    
    def _create_fire_systems(self) -> Dict:
        """Create fire system realism factors"""
        return {
            'sprinkler_economics': {
                'description': 'When sprinklers become economically justified',
                'cost_analysis': {
                    'installation': '$1.50/sq ft average',
                    'area_credit_value': 'Can add 25% area = worth $XX based on project',
                    'breakeven_project_size': '40000 sq ft (typical)'
                },
                'common_scenarios': [
                    {
                        'description': 'Type V-B undersized building',
                        'base_limits': '6000 sq ft',
                        'with_sprinklers': '7500 sq ft',
                        'cost_justification': 'Minimal ($9000 for 1500 sq ft value)',
                        'recommendation': 'Install sprinklers (high ROI)'
                    },
                    {
                        'description': 'Type II-B business building',
                        'base_limits': '65000 sq ft',
                        'with_sprinklers': '81000 sq ft',
                        'cost_justification': 'High ($22500 for 16000 sq ft value)',
                        'recommendation': 'Install sprinklers (strong ROI)'
                    }
                ]
            },
            'fire_alarm_integration': {
                'description': 'Fire alarm system considerations',
                'triggers_compliance': [
                    'H occupancies (above H-4)',
                    'I occupancies (institutional)',
                    'Large assembly (A-1, A-2)',
                    'Pull stations often required'
                ],
                'cost_factors': {
                    'base_system': '$0.50/sq ft',
                    'integration_with_sprinklers': 'Add $0.10-0.20/sq ft',
                    'monitoring': 'Add $50-200/month ongoing'
                }
            },
            'system_interactions': {
                'sprinklers_plus_fire_alarm': 'Combination is standard for large buildings',
                'high_rise_requirements': 'Voice alarm, smoke control also required',
                'renovation_trigger': 'When replacing systems, code updates apply'
            }
        }
    
    def export(self, output_path: str = "ground_truth/realism_upgrades.json") -> None:
        """Export upgrades to JSON"""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.upgrades, f, indent=2)
            print(f"✓ Realism upgrades exported to {output_path}")
        except Exception as e:
            print(f"✗ Error exporting upgrades: {e}")


def main():
    """Main entry point"""
    import sys
    
    try:
        upgrader = RealismUpgrader()
        upgrades = upgrader.generate_upgrades()
        
        print("\n" + "="*70)
        print("NBC REALISM UPGRADE SYSTEM")
        print("="*70)
        
        categories = list(upgrades.keys())
        print(f"\nGenerating {len(categories)} upgrade categories:")
        for cat in categories:
            print(f"  ✓ {cat}")
        
        # Export
        upgrader.export()
        
        print("\n" + "="*70)
        print("STATUS: Realism upgrades generated successfully")
        print("="*70)
        print(f"\nSystem now includes:")
        print("  • Cost efficiency factors (10 construction types)")
        print("  • Design preferences & constraints")
        print("  • Construction feasibility metrics")
        print("  • Common regulatory variance practices")
        print("  • Occupancy-specific guidance")
        print("  • Fire systems economics")
        
        return 0
    
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
