#!/usr/bin/env python3
"""
NBC 2018 Compliance Solver - Rule Compiler

Transforms extracted NBC rules (YAML) into:
1. Knowledge Graph (nodes + edges)
2. Constraint Programming formulation
3. Compliance scoring system
4. Rule validation engine
"""

import yaml
import json
from typing import Dict, List, Tuple, Any, Set, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import sys
from pathlib import Path


class RuleType(Enum):
    """Classification of rule types for solver"""
    HEIGHT_LIMIT = "height_limit"
    AREA_LIMIT = "area_limit"
    FIRE_RATING = "fire_rating"
    OCCUPANCY_RESTRICTION = "occupancy_restriction"
    EGRESS_REQUIREMENT = "egress_requirement"
    MATERIAL_REQUIREMENT = "material_requirement"
    ACCESSIBILITY_REQUIREMENT = "accessibility_requirement"
    INTERIOR_FINISH = "interior_finish"
    SEPARATION_REQUIREMENT = "separation_requirement"
    CONSTRUCTION_TYPE_RESTRICTION = "construction_type_restriction"


class ConstraintType(Enum):
    """Types of constraints for solver"""
    EQUALITY = "equality"              # X == value
    INEQUALITY_LT = "inequality_lt"    # X < value
    INEQUALITY_LTE = "inequality_lte"  # X <= value
    INEQUALITY_GT = "inequality_gt"    # X > value
    INEQUALITY_GTE = "inequality_gte"  # X >= value
    IMPLICATION = "implication"        # If X then Y
    OR_CONSTRAINT = "or_constraint"    # X OR Y
    AND_CONSTRAINT = "and_constraint"  # X AND Y


@dataclass
class KGNode:
    """Knowledge Graph Node"""
    node_id: str
    node_type: str                    # e.g., "occupancy", "construction_type", "property"
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class KGEdge:
    """Knowledge Graph Edge"""
    source_id: str
    target_id: str
    edge_type: str                    # e.g., "has_limit", "requires", "restricts"
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Constraint:
    """Constraint for satisfaction/solving"""
    constraint_id: str
    constraint_type: ConstraintType
    variables: List[str]              # Variable names involved
    operator: Optional[str] = None    # For inequalities like "<", ">", "=="
    value: Optional[Any] = None       # Right-hand side of constraint
    rule_references: List[str] = field(default_factory=list)  # NBC sections
    
    def to_dict(self) -> Dict:
        return {
            "constraint_id": self.constraint_id,
            "constraint_type": self.constraint_type.value,
            "variables": self.variables,
            "operator": self.operator,
            "value": self.value,
            "rule_references": self.rule_references
        }


class NBCRuleCompiler:
    """Compiles NBC rules from YAML into Knowledge Graph + Constraints"""
    
    def __init__(self, yaml_path: str):
        """Initialize compiler and load rules"""
        self.yaml_path = Path(yaml_path)
        self.rules = self._load_yaml()
        
        # Knowledge Graph structures
        self.nodes: Dict[str, KGNode] = {}
        self.edges: Dict[str, List[KGEdge]] = {}
        self.constraints: List[Constraint] = []
        
        # Tracking
        self.node_counter = 0
        self.edge_counter = 0
        self.constraint_counter = 0
        
        # Validation state
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
        
    def _load_yaml(self) -> Dict:
        """Load ground truth YAML file"""
        try:
            with open(self.yaml_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Ground truth file not found: {self.yaml_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML file: {e}")
    
    def compile(self) -> Tuple[Dict[str, KGNode], List[KGEdge], List[Constraint]]:
        """Main compilation pipeline"""
        print("Starting NBC rule compilation...")
        
        # Phase 1: Create KG nodes from occupancy groups
        self._compile_occupancy_nodes()
        print(f"✓ Created {len([n for n in self.nodes.values() if n.node_type == 'occupancy'])} occupancy nodes")
        
        # Phase 2: Create KG nodes from construction types
        self._compile_construction_type_nodes()
        print(f"✓ Created {len([n for n in self.nodes.values() if n.node_type == 'construction_type'])} construction type nodes")
        
        # Phase 3: Create property nodes (height, area, fire rating, etc.)
        self._compile_property_nodes()
        print(f"✓ Created {len([n for n in self.nodes.values() if n.node_type == 'property'])} property nodes")
        
        # Phase 4: Create edges connecting rules
        self._compile_edges()
        edge_count = sum(len(edges) for edges in self.edges.values())
        print(f"✓ Created {edge_count} knowledge graph edges")
        
        # Phase 5: Generate constraints for solver
        self._compile_constraints()
        print(f"✓ Generated {len(self.constraints)} solver constraints")
        
        # Phase 6: Validation
        self._validate_rules()
        
        return self.nodes, self._flatten_edges(), self.constraints
    
    def _compile_occupancy_nodes(self) -> None:
        """Create KG nodes for each occupancy group and subgroup"""
        occupancy_groups = self.rules.get('occupancy_groups', {})
        
        for group_code, group_data in occupancy_groups.items():
            # Create main occupancy node
            node_id = f"occupancy_{group_code}"
            node = KGNode(
                node_id=node_id,
                node_type="occupancy",
                label=group_data.get('name', group_code),
                properties={
                    'code': group_code,
                    'description': group_data.get('description', ''),
                    'has_subgroups': 'subgroups' in group_data
                }
            )
            self.nodes[node_id] = node
            
            # Create subgroup nodes if they exist
            subgroups = group_data.get('subgroups', {})
            for sub_code, sub_data in subgroups.items():
                sub_node_id = f"occupancy_{group_code}_{sub_code}"
                sub_node = KGNode(
                    node_id=sub_node_id,
                    node_type="occupancy",
                    label=sub_data.get('name', f"{group_code}-{sub_code}"),
                    properties={
                        'code': sub_code,
                        'parent_group': group_code,
                        'description': sub_data.get('description', ''),
                        'capacity_threshold': sub_data.get('capacity_threshold'),
                        'examples': sub_data.get('examples', [])
                    }
                )
                self.nodes[sub_node_id] = sub_node
    
    def _compile_construction_type_nodes(self) -> None:
        """Create KG nodes for construction types"""
        construction_types = self.rules.get('construction_types', {})
        
        for type_code, type_data in construction_types.items():
            node_id = f"construction_type_{type_code}"
            node = KGNode(
                node_id=node_id,
                node_type="construction_type",
                label=type_data.get('name', type_code),
                properties={
                    'code': type_code,
                    'description': type_data.get('name', ''),
                    'fire_rating_structural_frame': type_data.get('fire_rating_structural_frame'),
                    'fire_rating_exterior_walls': type_data.get('fire_rating_exterior_walls'),
                    'fire_rating_interior_bearing_walls': type_data.get('fire_rating_interior_bearing_walls'),
                    'primary_materials': type_data.get('primary_materials', [])
                }
            )
            self.nodes[node_id] = node
    
    def _compile_property_nodes(self) -> None:
        """Create nodes for properties like height limits, area limits, fire ratings"""
        # Height limit properties
        node = KGNode(
            node_id="property_height_limit",
            node_type="property",
            label="Height Limit (feet)",
            properties={'unit': 'feet', 'category': 'dimensional'}
        )
        self.nodes[node.node_id] = node
        
        # Area limit properties
        node = KGNode(
            node_id="property_area_limit",
            node_type="property",
            label="Area Limit (sq ft)",
            properties={'unit': 'square_feet', 'category': 'dimensional'}
        )
        self.nodes[node.node_id] = node
        
        # Fire rating properties
        node = KGNode(
            node_id="property_fire_rating",
            node_type="property",
            label="Fire Resistance Rating",
            properties={'unit': 'hours', 'category': 'fire_safety'}
        )
        self.nodes[node.node_id] = node
        
        # Story limit properties
        node = KGNode(
            node_id="property_story_limit",
            node_type="property",
            label="Story Limit",
            properties={'unit': 'stories', 'category': 'dimensional'}
        )
        self.nodes[node.node_id] = node
        
        # Occupancy load properties
        node = KGNode(
            node_id="property_occupancy_load",
            node_type="property",
            label="Occupancy Load",
            properties={'unit': 'persons', 'category': 'capacity'}
        )
        self.nodes[node.node_id] = node
    
    def _compile_edges(self) -> None:
        """Create edges connecting rules (relationships between properties and restrictions)"""
        # Edges: construction_type -> has_limit -> property_height_limit
        construction_types = self.rules.get('construction_types', {})
        for type_code in construction_types.keys():
            src = f"construction_type_{type_code}"
            
            # Height limit edge
            edge = KGEdge(
                source_id=src,
                target_id="property_height_limit",
                edge_type="has_limit",
                label=f"{type_code} has height limit",
                properties={'rule_section': 'Table 504.3'}
            )
            self._add_edge(edge)
            
            # Fire rating edge
            edge = KGEdge(
                source_id=src,
                target_id="property_fire_rating",
                edge_type="requires",
                label=f"{type_code} requires fire rating",
                properties={'rule_section': 'Chapter 6'}
            )
            self._add_edge(edge)
        
        # Edges: occupancy -> has_requirement -> property_*
        occupancy_groups = self.rules.get('occupancy_groups', {})
        for group_code in occupancy_groups.keys():
            src = f"occupancy_{group_code}"
            
            # Height limit
            edge = KGEdge(
                source_id=src,
                target_id="property_height_limit",
                edge_type="has_limit",
                label=f"{group_code} occupancy subject to height limits",
                properties={'rule_section': 'Table 504.3'}
            )
            self._add_edge(edge)
            
            # Area limit
            edge = KGEdge(
                source_id=src,
                target_id="property_area_limit",
                edge_type="has_limit",
                label=f"{group_code} occupancy subject to area limits",
                properties={'rule_section': 'Table 506.2'}
            )
            self._add_edge(edge)
        
        # Edges: fire-resistance separations between occupancy pairs
        occupancy_separations = self.rules.get('fire_resistance_requirements', {}).get('occupancy_separations', {})
        for separation_key, rating in occupancy_separations.items():
            if '_vs_' in separation_key:
                occupancies = separation_key.split('_vs_')
                src = f"occupancy_{occupancies[0]}"
                tgt = f"occupancy_{occupancies[1]}"
                
                edge = KGEdge(
                    source_id=src,
                    target_id=tgt,
                    edge_type="separation_requires",
                    label=f"{occupancies[0]} vs {occupancies[1]}: {rating}-hr rating",
                    properties={
                        'fire_rating_hours': rating,
                        'rule_section': 'Table 508.4'
                    }
                )
                self._add_edge(edge)
    
    def _add_edge(self, edge: KGEdge) -> None:
        """Add edge to knowledge graph"""
        key = edge.source_id
        if key not in self.edges:
            self.edges[key] = []
        self.edges[key].append(edge)
    
    def _flatten_edges(self) -> List[KGEdge]:
        """Flatten edges dictionary to list"""
        all_edges = []
        for edges_list in self.edges.values():
            all_edges.extend(edges_list)
        return all_edges
    
    def _compile_constraints(self) -> None:
        """Generate constraints in CSP/ILP form"""
        height_limits = self.rules.get('height_area_limitations', {}).get('height_limits_feet', {})
        area_limits = self.rules.get('height_area_limitations', {}).get('area_limits_sqft', {})
        
        # Generate height limit constraints
        for const_type, occupancies in height_limits.items():
            for occ_code, limit in occupancies.items():
                if limit is not None:
                    constraint_id = f"constraint_height_{const_type}_{occ_code}"
                    constraint = Constraint(
                        constraint_id=constraint_id,
                        constraint_type=ConstraintType.INEQUALITY_LTE,
                        variables=["building_height_feet"],
                        operator="<=",
                        value=limit,
                        rule_references=[f"Table 504.3 {const_type}/{occ_code}"]
                    )
                    self.constraints.append(constraint)
        
        # Generate area limit constraints (simplified)
        for const_type, occupancies in area_limits.items():
            for occ_code, limit in occupancies.items():
                if limit is not None:
                    constraint_id = f"constraint_area_{const_type}_{occ_code}"
                    constraint = Constraint(
                        constraint_id=constraint_id,
                        constraint_type=ConstraintType.INEQUALITY_LTE,
                        variables=["building_area_sqft"],
                        operator="<=",
                        value=limit,
                        rule_references=[f"Table 506.2 {const_type}/{occ_code}"]
                    )
                    self.constraints.append(constraint)
        
        # Fire rating constraints from separations
        occupancy_separations = self.rules.get('fire_resistance_requirements', {}).get('occupancy_separations', {})
        for sep_key, rating in occupancy_separations.items():
            if rating > 0:
                constraint_id = f"constraint_fire_rating_{sep_key}"
                constraint = Constraint(
                    constraint_id=constraint_id,
                    constraint_type=ConstraintType.INEQUALITY_GTE,
                    variables=["separation_fire_rating_hours"],
                    operator=">=",
                    value=rating,
                    rule_references=[f"Table 508.4 {sep_key}"]
                )
                self.constraints.append(constraint)
    
    def _validate_rules(self) -> None:
        """Validate rule consistency and completeness"""
        # Check for orphaned nodes
        all_nodes_referenced = set()
        for edges_list in self.edges.values():
            for edge in edges_list:
                all_nodes_referenced.add(edge.source_id)
                all_nodes_referenced.add(edge.target_id)
        
        orphaned_nodes = set(self.nodes.keys()) - all_nodes_referenced
        if orphaned_nodes:
            self.validation_warnings.append(f"Orphaned nodes: {orphaned_nodes}")
        
        # Check for consistent fire ratings across construction types
        const_types = self.rules.get('construction_types', {})
        for type_code, type_data in const_types.items():
            frame_rating = type_data.get('fire_rating_structural_frame')
            wall_rating = type_data.get('fire_rating_exterior_walls')
            bearing_rating = type_data.get('fire_rating_interior_bearing_walls')
            
            if frame_rating > wall_rating or frame_rating > bearing_rating:
                self.validation_warnings.append(
                    f"Type {type_code}: Structural frame rating ({frame_rating}) "
                    f"exceeds wall ({wall_rating}) or bearing ({bearing_rating}) ratings"
                )
    
    def export_knowledge_graph(self, output_path: str) -> None:
        """Export knowledge graph to JSON"""
        graph_data = {
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'edges': [edge.to_dict() for edge in self._flatten_edges()],
            'metadata': {
                'node_count': len(self.nodes),
                'edge_count': len(self._flatten_edges()),
                'source': str(self.yaml_path)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        print(f"Knowledge graph exported to {output_path}")
    
    def export_constraints(self, output_path: str) -> None:
        """Export solver constraints to JSON"""
        constraints_data = {
            'constraints': [c.to_dict() for c in self.constraints],
            'metadata': {
                'constraint_count': len(self.constraints),
                'types': list(set(c.constraint_type.value for c in self.constraints))
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(constraints_data, f, indent=2)
        print(f"Constraints exported to {output_path}")
    
    def export_validation_report(self, output_path: str) -> None:
        """Export validation report"""
        report = {
            'compilation_status': 'SUCCESS' if not self.validation_errors else 'FAILED',
            'nodes_created': len(self.nodes),
            'edges_created': len(self._flatten_edges()),
            'constraints_created': len(self.constraints),
            'errors': self.validation_errors,
            'warnings': self.validation_warnings
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Validation report exported to {output_path}")


def main():
    """Main entry point"""
    yaml_path = "ground_truth/ground_truth.yml"
    
    try:
        compiler = NBCRuleCompiler(yaml_path)
        nodes, edges, constraints = compiler.compile()
        
        # Export results
        compiler.export_knowledge_graph("ground_truth/knowledge_graph.json")
        compiler.export_constraints("ground_truth/constraints.json")
        compiler.export_validation_report("ground_truth/validation_report.json")
        
        print("\n" + "="*60)
        print("COMPILATION SUCCESSFUL")
        print("="*60)
        print(f"Nodes: {len(nodes)}")
        print(f"Edges: {len(edges)}")
        print(f"Constraints: {len(constraints)}")
        
        if compiler.validation_warnings:
            print(f"\nWarnings: {len(compiler.validation_warnings)}")
            for warning in compiler.validation_warnings:
                print(f"  - {warning}")
        
        return 0
    
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
