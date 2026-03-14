from pathlib import Path
from uuid import uuid4
from typing import List, Tuple

from graph.connectivity import is_fully_connected
from graph.manhattan_path import max_travel_distance


class OntologyBridge:
    def __init__(self, ontology_file, reasoner_mode="try"):
        self.ontology_file = ontology_file
        self.reasoner_mode = reasoner_mode  # off | try | require
        self.owlready_available = False
        self.load_error = None
        self.onto = None
        self.owl = None
        self.namespace = None

        try:
            import owlready2 as owl

            self.owl = owl
            self.owlready_available = True
        except Exception as exc:
            self.load_error = str(exc)
            return

        try:
            self._load_or_bootstrap_ontology()
        except Exception as exc:
            self.load_error = str(exc)

    def _load_or_bootstrap_ontology(self):
        onto_path = Path(self.ontology_file)
        onto_path.parent.mkdir(parents=True, exist_ok=True)

        # Keep loading deterministic and quiet: use file path first; if that
        # fails, bootstrap from the canonical in-memory IRI.
        if onto_path.exists() and onto_path.stat().st_size > 0:
            try:
                self.onto = self.owl.get_ontology(str(onto_path.resolve())).load()
            except Exception as exc:
                self.load_error = str(exc)
                self.onto = self.owl.get_ontology("http://genai.local/regulatory.owl")
        else:
            self.onto = self.owl.get_ontology("http://genai.local/regulatory.owl")

        self.namespace = self.onto
        # Always ensure schema and SWRL rules are present after load/bootstrap.
        self._ensure_schema_and_rules()
        self.onto.save(file=str(onto_path), format="rdfxml")

    def _ensure_class(self, name, base):
        existing = getattr(self.namespace, name, None)
        if existing is not None:
            return existing
        with self.onto:
            return type(name, (base,), {})

    def _ensure_object_property(self, name, domain_cls, range_cls):
        prop = getattr(self.namespace, name, None)
        if prop is None:
            with self.onto:
                prop = type(name, (self.owl.ObjectProperty,), {})
        if domain_cls is not None and domain_cls not in prop.domain:
            prop.domain.append(domain_cls)
        if range_cls is not None and range_cls not in prop.range:
            prop.range.append(range_cls)
        return prop

    def _ensure_data_property(self, name, domain_cls, range_type):
        prop = getattr(self.namespace, name, None)
        if prop is None:
            with self.onto:
                prop = type(name, (self.owl.DataProperty,), {})
        if domain_cls is not None and domain_cls not in prop.domain:
            prop.domain.append(domain_cls)
        if range_type is not None and range_type not in prop.range:
            prop.range.append(range_type)
        return prop

    def _ensure_swrl_rule(self, rule_name, rule_text):
        rule = getattr(self.namespace, rule_name, None)
        with self.onto:
            if rule is None:
                rule = self.owl.Imp(rule_name)
            rule.set_as_rule(rule_text)

    def _ensure_schema_and_rules(self):
        Building = self._ensure_class("Building", self.owl.Thing)
        ResidentialBuilding = self._ensure_class("ResidentialBuilding", Building)
        Room = self._ensure_class("Room", self.owl.Thing)
        HabitableRoom = self._ensure_class("HabitableRoom", Room)
        Bedroom = self._ensure_class("Bedroom", HabitableRoom)
        LivingRoom = self._ensure_class("LivingRoom", HabitableRoom)
        Kitchen = self._ensure_class("Kitchen", Room)
        Bathroom = self._ensure_class("Bathroom", Room)
        WC = self._ensure_class("WC", Room)
        Door = self._ensure_class("Door", self.owl.Thing)
        Exit = self._ensure_class("Exit", self.owl.Thing)
        Corridor = self._ensure_class("Corridor", self.owl.Thing)

        Violation = self._ensure_class("Violation", self.owl.Thing)
        self._ensure_class("MinAreaViolation", Violation)
        self._ensure_class("TravelDistanceViolation", Violation)
        self._ensure_class("ConnectivityViolation", Violation)
        self._ensure_class("ExitWidthViolation", Violation)
        self._ensure_class("CorridorWidthViolation", Violation)

        AreaNonCompliantRoom = self._ensure_class("AreaNonCompliantRoom", Room)
        TravelDistanceNonCompliantBuilding = self._ensure_class("TravelDistanceNonCompliantBuilding", Building)
        ExitWidthNonCompliantExit = self._ensure_class("ExitWidthNonCompliantExit", Exit)

        self._ensure_object_property("containsRoom", Building, Room)
        self._ensure_object_property("hasCorridor", Building, Corridor)
        self._ensure_object_property("hasDoor", Room, Door)
        self._ensure_object_property("connects", Door, Room)
        self._ensure_object_property("hasExit", Building, Exit)
        self._ensure_object_property("hasViolation", Building, Violation)

        self._ensure_data_property("hasArea", Room, float)
        self._ensure_data_property("hasMinRequiredArea", Room, float)
        self._ensure_data_property("hasOccupantLoad", Building, float)
        self._ensure_data_property("hasMaxTravelDistance", Building, float)
        self._ensure_data_property("hasTravelDistance", Building, float)
        self._ensure_data_property("hasDoorWidth", Door, float)
        self._ensure_data_property("hasExitWidth", Exit, float)
        self._ensure_data_property("hasMinExitWidth", Exit, float)
        self._ensure_data_property("hasCorridorWidth", Corridor, float)
        self._ensure_data_property("hasMinCorridorWidth", Corridor, float)
        self._ensure_data_property("hasViolationCode", Violation, str)
        self._ensure_data_property("hasViolationMessage", Violation, str)
        self._ensure_data_property("hasViolationEntity", Violation, str)

        self._ensure_swrl_rule(
            "Rule_MinArea_NonCompliant",
            "Room(?r), hasArea(?r, ?a), hasMinRequiredArea(?r, ?m), lessThan(?a, ?m) -> AreaNonCompliantRoom(?r)",
        )
        self._ensure_swrl_rule(
            "Rule_TravelDistance_NonCompliant",
            "Building(?b), hasTravelDistance(?b, ?d), hasMaxTravelDistance(?b, ?m), greaterThan(?d, ?m) -> TravelDistanceNonCompliantBuilding(?b)",
        )
        self._ensure_swrl_rule(
            "Rule_ExitWidth_NonCompliant",
            "Exit(?e), hasExitWidth(?e, ?w), hasMinExitWidth(?e, ?m), lessThan(?w, ?m) -> ExitWidthNonCompliantExit(?e)",
        )
        self._ensure_swrl_rule(
            "Rule_CorridorWidth_NonCompliant",
            "Corridor(?c), hasCorridorWidth(?c, ?w), hasMinCorridorWidth(?c, ?m), lessThan(?w, ?m) -> CorridorWidthViolation(?c)",
        )

    def _safe_class(self, name, fallback_name):
        cls = getattr(self.namespace, name, None)
        if cls is None:
            cls = getattr(self.namespace, fallback_name, None)
        return cls

    def _new_violation(self, code, entity, message):
        return {
            "code": code,
            "entity": entity,
            "message": message,
        }

    def get_allowed_room_types(self):
        defaults = {"Bedroom", "LivingRoom", "Kitchen", "Bathroom", "WC"}
        if not self.owlready_available or self.onto is None:
            return defaults

        try:
            allowed = set()
            for name in defaults:
                cls = getattr(self.namespace, name, None)
                if cls is not None:
                    allowed.add(name)
            return allowed or defaults
        except Exception:
            return defaults

    # ── KG-driven generation intents ──────────────────────────────────────

    def get_adjacency_intents(self, room_types=None):
        """Return weighted adjacency pairs derived from the ontology.

        Falls back to a hard-coded graph when OWL classes don't carry
        adjacency properties.  The intent is that this method *replaces*
        the static list in ``adjacency_intent.py`` once KG annotations are
        richer.

        Returns list of (type_a, type_b, weight).
        """
        # OWL-derived intents would query datatype properties here.
        # For now, infer from ontology class hierarchy + domain knowledge:
        intents = [
            ("Kitchen",    "DiningRoom",  1.0),
            ("Kitchen",    "LivingRoom",  0.9),
            ("LivingRoom", "DiningRoom",  0.7),
            ("LivingRoom", "Bedroom",     0.4),
            ("Bedroom",    "Bathroom",    0.8),
            ("Bedroom",    "WC",          0.6),
            ("Bedroom",    "DressingArea",0.5),
            ("LivingRoom", "Lobby",       0.6),
            ("Lobby",      "Kitchen",     0.3),
        ]

        if room_types:
            type_set = set(room_types)
            intents = [(a, b, w) for a, b, w in intents
                       if a in type_set or b in type_set]

        return intents

    def get_intent_edges(self, room_types=None) -> List[Tuple[str, str, float]]:
        """Public KG intent API used by generation/re-ranking layers."""
        return self.get_adjacency_intents(room_types=room_types)

    def get_zone_rules(self, room_types=None):
        """Return public / service / private zone mapping for room types.

        Inference: rooms that appear as range of ``containsRoom`` from a
        ``ResidentialBuilding`` and inherit from HabitableRoom are public/
        service. This is a simplified procedural mapping pending richer OWL
        annotations.

        Returns dict {room_type: zone}.
        """
        zone_map = {
            "LivingRoom":  "public",
            "DrawingRoom": "public",
            "Lobby":       "public",
            "DiningRoom":  "public",
            "Kitchen":     "service",
            "Laundry":     "service",
            "Store":       "service",
            "Storage":     "service",
            "Bedroom":     "private",
            "Bathroom":    "private",
            "WC":          "private",
            "DressingArea":"private",
            "Study":       "private",
            "PrayerRoom":  "private",
            "Garage":      "service",
            "Stairs":      "service",
            "Staircase":   "service",
            "Passage":     "service",
            "Lawn":        "public",
            "OpenSpace":   "public",
            "Balcony":     "public",
            "SideGarden":  "public",
            "Backyard":    "public",
        }

        if room_types:
            return {rt: zone_map.get(rt, "service") for rt in room_types}
        return dict(zone_map)

    def validate_spec_semantics(self, spec):
        """
        KG semantic gate before layout generation.
        Ensures room types and key occupancy semantics are ontology-aligned.
        """
        allowed_types = self.get_allowed_room_types()
        errors = []
        warnings = []

        occupancy = spec.get("occupancy", "Residential")
        if occupancy != "Residential":
            errors.append(f"Ontology currently supports Residential flow, got '{occupancy}'")

        rooms = spec.get("rooms", [])
        if not rooms:
            errors.append("No rooms in spec for KG validation")

        for idx, room in enumerate(rooms):
            room_type = room.get("type")
            if room_type not in allowed_types:
                errors.append(f"rooms[{idx}] has invalid ontology room type '{room_type}'")

        if not any(r.get("type") == "LivingRoom" for r in rooms):
            warnings.append("No LivingRoom specified; public-zone semantics may degrade")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "ontology_loaded": bool(self.onto is not None),
            "owlready_available": self.owlready_available,
            "reasoner_available": self.owlready_available,
            "allowed_room_types": sorted(allowed_types),
        }

    def _procedural_violations(self, building, max_allowed_travel, travel_distance, min_exit_width, connected):
        violations = []
        for room in building.rooms:
            if room.final_area is not None and room.min_area is not None and room.final_area < room.min_area:
                violations.append(
                    self._new_violation(
                        "MIN_AREA",
                        room.name,
                        f"{room.name} has area {room.final_area} below required {room.min_area}",
                    )
                )

        if travel_distance > max_allowed_travel:
            violations.append(
                self._new_violation(
                    "TRAVEL_DISTANCE",
                    "Building",
                    f"Max travel distance {travel_distance} exceeds allowed {max_allowed_travel}",
                )
            )

        if building.exit and building.exit.width < min_exit_width:
            violations.append(
                self._new_violation(
                    "EXIT_WIDTH",
                    "Exit",
                    f"Exit width {building.exit.width} is below minimum {min_exit_width}",
                )
            )

        if not connected:
            violations.append(
                self._new_violation(
                    "CONNECTIVITY",
                    "Building",
                    "Layout graph is not fully connected",
                )
            )

        return violations

    def validate(self, building, rule_engine):
        max_allowed_travel = rule_engine.get_max_travel_distance(building.occupancy_type)
        connected = is_fully_connected(building)
        travel_distance = max_travel_distance(building)
        min_exit_width = rule_engine.data[building.occupancy_type]["exit"]["min_width"]

        if self.reasoner_mode == "off":
            violations = self._procedural_violations(
                building,
                max_allowed_travel,
                travel_distance,
                min_exit_width,
                connected,
            )
            return {
                "reasoner": "off",
                "reasoner_success": False,
                "reasoner_error": None,
                "ontology_loaded": bool(self.onto is not None),
                "ontology_file": self.ontology_file,
                "load_error": self.load_error,
                "max_allowed_travel_distance": max_allowed_travel,
                "travel_distance": travel_distance,
                "violations": violations,
                "valid": len(violations) == 0,
            }

        if not self.owlready_available or self.onto is None:
            violations = self._procedural_violations(
                building,
                max_allowed_travel,
                travel_distance,
                min_exit_width,
                connected,
            )
            return {
                "reasoner": "not-configured",
                "reasoner_success": False,
                "reasoner_error": self.load_error,
                "ontology_loaded": False,
                "ontology_file": self.ontology_file,
                "load_error": self.load_error,
                "max_allowed_travel_distance": max_allowed_travel,
                "travel_distance": travel_distance,
                "violations": violations,
                "valid": len(violations) == 0,
            }

        try:
            self._load_or_bootstrap_ontology()

            run_id = uuid4().hex[:8]
            building_cls_name = f"{building.occupancy_type}Building"
            building_cls = self._safe_class(building_cls_name, "Building")
            building_individual = building_cls(f"Building_{run_id}")
            building_individual.hasOccupantLoad.append(float(building.occupant_load))
            building_individual.hasMaxTravelDistance.append(float(max_allowed_travel))
            building_individual.hasTravelDistance.append(float(travel_distance))

            room_type_map = {
                "Bedroom": "Bedroom",
                "LivingRoom": "LivingRoom",
                "Kitchen": "Kitchen",
                "Bathroom": "Bathroom",
                "WC": "WC",
            }

            room_individual_map = {}
            for idx, room in enumerate(building.rooms, start=1):
                cls_name = room_type_map.get(room.room_type, "Room")
                room_cls = self._safe_class(cls_name, "Room")
                room_individual = room_cls(f"Room_{run_id}_{idx}_{room.room_type}")
                room_individual.hasArea.append(float(room.final_area))
                if room.min_area is not None:
                    room_individual.hasMinRequiredArea.append(float(room.min_area))
                building_individual.containsRoom.append(room_individual)
                room_individual_map[room.name] = room_individual

            for idx, door in enumerate(building.doors, start=1):
                door_cls = self._safe_class("Door", "Thing")
                door_individual = door_cls(f"Door_{run_id}_{idx}")
                door_individual.hasDoorWidth.append(float(door.width))
                room_a = room_individual_map.get(door.room_a.name)
                room_b = room_individual_map.get(door.room_b.name)
                if room_a is not None:
                    room_a.hasDoor.append(door_individual)
                    door_individual.connects.append(room_a)
                if room_b is not None:
                    room_b.hasDoor.append(door_individual)
                    door_individual.connects.append(room_b)

            if building.exit is not None:
                exit_cls = self._safe_class("Exit", "Thing")
                exit_individual = exit_cls(f"Exit_{run_id}")
                exit_individual.hasExitWidth.append(float(building.exit.width))
                exit_individual.hasMinExitWidth.append(float(min_exit_width))
                building_individual.hasExit.append(exit_individual)
            else:
                exit_individual = None

            # Instantiate corridor individuals
            corridor_cls = self._safe_class("Corridor", "Thing")
            min_corridor_width = 1.2  # Indian NBC 2016 residential default
            corridor_individual_map = {}
            for idx, corridor in enumerate(getattr(building, 'corridors', []), start=1):
                corr_ind = corridor_cls(f"Corridor_{run_id}_{idx}")
                corr_ind.hasCorridorWidth.append(float(corridor.width))
                corr_ind.hasMinCorridorWidth.append(float(min_corridor_width))
                building_individual.hasCorridor.append(corr_ind)
                corridor_individual_map[corridor.name] = corr_ind

            reasoner_success = False
            reasoner_error = None
            reasoner_used = "owlready2"
            try:
                self.owl.sync_reasoner_pellet(
                    [self.onto],
                    infer_property_values=True,
                    infer_data_property_values=True,
                    debug=0,
                )
                reasoner_success = True
                reasoner_used = "owlready2-pellet"
            except Exception as pellet_error:
                reasoner_error = str(pellet_error)
                try:
                    self.owl.sync_reasoner([self.onto], infer_property_values=True, debug=0)
                    reasoner_success = True
                    reasoner_used = "owlready2-hermit"
                    reasoner_error = None
                except Exception as sync_error:
                    reasoner_error = str(sync_error)

            if reasoner_success:
                area_non_compliant_cls = self._safe_class("AreaNonCompliantRoom", "Room")
                travel_non_compliant_cls = self._safe_class("TravelDistanceNonCompliantBuilding", "Building")
                exit_non_compliant_cls = self._safe_class("ExitWidthNonCompliantExit", "Exit")

                violations = []
                for room_name, room_individual in room_individual_map.items():
                    if area_non_compliant_cls in room_individual.is_a:
                        final_area = next(iter(room_individual.hasArea), None)
                        min_area = next(iter(room_individual.hasMinRequiredArea), None)
                        violations.append(
                            self._new_violation(
                                "MIN_AREA",
                                room_name,
                                f"{room_name} has area {final_area} below required {min_area}",
                            )
                        )

                if travel_non_compliant_cls in building_individual.is_a:
                    violations.append(
                        self._new_violation(
                            "TRAVEL_DISTANCE",
                            "Building",
                            f"Max travel distance {travel_distance} exceeds allowed {max_allowed_travel}",
                        )
                    )

                if exit_individual is not None and exit_non_compliant_cls in exit_individual.is_a:
                    violations.append(
                        self._new_violation(
                            "EXIT_WIDTH",
                            "Exit",
                            f"Exit width {building.exit.width} is below minimum {min_exit_width}",
                        )
                    )

                if not connected:
                    violations.append(
                        self._new_violation(
                            "CONNECTIVITY",
                            "Building",
                            "Layout graph is not fully connected",
                        )
                    )
            else:
                violations = self._procedural_violations(
                    building,
                    max_allowed_travel,
                    travel_distance,
                    min_exit_width,
                    connected,
                )

            if self.reasoner_mode == "require" and not reasoner_success:
                violations.append(
                    self._new_violation(
                        "REASONER_REQUIRED",
                        "OntologyBridge",
                        f"Reasoner was required but failed: {reasoner_error}",
                    )
                )

            return {
                "reasoner": reasoner_used,
                "reasoner_success": reasoner_success,
                "reasoner_error": reasoner_error,
                "ontology_loaded": True,
                "ontology_file": self.ontology_file,
                "load_error": self.load_error,
                "max_allowed_travel_distance": max_allowed_travel,
                "travel_distance": travel_distance,
                "violations": violations,
                "valid": len(violations) == 0,
            }
        except Exception as exc:
            return {
                "reasoner": "ontology-bridge-error",
                "reasoner_success": False,
                "reasoner_error": str(exc),
                "ontology_loaded": self.onto is not None,
                "ontology_file": self.ontology_file,
                "load_error": self.load_error,
                "max_allowed_travel_distance": max_allowed_travel,
                "travel_distance": travel_distance,
                "violations": [
                    {
                        "code": "ONTOLOGY_ERROR",
                        "entity": "OntologyBridge",
                        "message": str(exc),
                    }
                ],
                "valid": False,
            }