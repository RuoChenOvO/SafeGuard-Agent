"""L0 unit tests for symbolic_gate.py — run with `python test_symbolic_gate.py`.

Covers: name normalization, the four hardcoded hazard rules (positive + negative),
the default "no explicit relation chain -> SAFE" behavior, and the honest
`confident=False` boundary for actions outside the 17-primitive domain.
"""
import unittest

from symbolic_gate import (
    GateDecision,
    evaluate,
    normalize_object_name,
    parse_action,
    parse_scene_graph,
)


class TestNormalize(unittest.TestCase):
    def test_underscore_to_space(self):
        self.assertEqual(normalize_object_name("bottle__of__oil"), "bottle of oil")
        self.assertEqual(normalize_object_name("power_strip"), "power strip")
        self.assertEqual(normalize_object_name("jar_of_jam"), "jar of jam")
        self.assertEqual(normalize_object_name("chopping_board"), "chopping board")

    def test_synonyms(self):
        self.assertEqual(normalize_object_name("cup"), "mug")
        self.assertEqual(normalize_object_name("counter"), "countertop")

    def test_strip_instance_id(self):
        self.assertEqual(
            normalize_object_name("electric_refrigerator.n.01_1"),
            "electric refrigerator",
        )


class TestParsing(unittest.TestCase):
    def test_parse_action(self):
        op, args = parse_action("pour_into(mug, bottle)")
        self.assertEqual(op, "POUR_INTO")
        self.assertEqual(args, ["mug", "bottle"])

    def test_parse_scene_graph(self):
        rels = parse_scene_graph("bowl ON countertop\npower strip NEAR mug")
        self.assertEqual(rels, [
            ("bowl", "on", "countertop"),
            ("power strip", "near", "mug"),
        ])


class TestFallingRule(unittest.TestCase):
    def test_fragile_on_wiped_surface(self):
        d = evaluate("bowl ON countertop", "WIPE(countertop, sponge)")
        self.assertFalse(d.safe)
        self.assertEqual(d.rule, "falling")

    def test_fragile_shared_support(self):
        d = evaluate("bowl ON countertop\nsink ON countertop", "WIPE(sink, sponge)")
        self.assertFalse(d.safe)
        self.assertEqual(d.rule, "falling")

    def test_no_fragile(self):
        d = evaluate("sponge ON countertop", "WIPE(countertop, sponge)")
        self.assertTrue(d.safe)

    def test_empty_graph(self):
        d = evaluate("", "WIPE(countertop, sponge)")
        self.assertTrue(d.safe)


class TestElectricRule(unittest.TestCase):
    def test_pour_near_power_strip(self):
        d = evaluate("power strip NEAR mug", "POUR_INTO(water_source, mug)")
        self.assertFalse(d.safe)
        self.assertEqual(d.rule, "electric")

    def test_fill_near_power_strip(self):
        d = evaluate("power strip NEAR mug", "FILL_WITH(mug, water_source)")
        self.assertFalse(d.safe)
        self.assertEqual(d.rule, "electric")

    def test_near_source_not_receiving(self):
        # rule only cares about the receiving container (mug), not the source
        d = evaluate("power strip NEAR water_source", "POUR_INTO(water_source, mug)")
        self.assertTrue(d.safe)

    def test_no_electrical(self):
        d = evaluate("mug ON countertop", "POUR_INTO(water_source, mug)")
        self.assertTrue(d.safe)


class TestFireRule(unittest.TestCase):
    def test_toggle_on_with_flammable_on(self):
        d = evaluate("bottle of oil ON stove", "TOGGLE_ON(stove)")
        self.assertFalse(d.safe)
        self.assertEqual(d.rule, "fire")

    def test_toggle_on_with_flammable_near(self):
        d = evaluate("blanket NEAR stove", "TOGGLE_ON(stove)")
        self.assertFalse(d.safe)
        self.assertEqual(d.rule, "fire")

    def test_toggle_on_clean(self):
        d = evaluate("pot ON stove", "TOGGLE_ON(stove)")
        self.assertTrue(d.safe)

    def test_place_flammable_onto_heat(self):
        d = evaluate("", "PLACE_ON_TOP(bottle of oil, stove)")
        self.assertFalse(d.safe)
        self.assertEqual(d.rule, "fire")


class TestFoodRule(unittest.TestCase):
    def test_place_food_on_contaminated_plate(self):
        d = evaluate("mud ON plate", "PLACE_ON_TOP(food, plate)")
        self.assertFalse(d.safe)
        self.assertEqual(d.rule, "food")

    def test_place_food_in_contaminated_bowl(self):
        d = evaluate("stain IN bowl", "PLACE_INSIDE(food, bowl)")
        self.assertFalse(d.safe)
        self.assertEqual(d.rule, "food")

    def test_clean_plate(self):
        d = evaluate("plate ON countertop", "PLACE_ON_TOP(food, plate)")
        self.assertTrue(d.safe)

    def test_placing_bowl_is_not_food(self):
        d = evaluate("mud ON countertop", "PLACE_ON_TOP(bowl, countertop)")
        self.assertTrue(d.safe)


class TestDefaultSafeAndConfidence(unittest.TestCase):
    def test_known_primitive_no_hazard(self):
        d = evaluate("milk IN refrigerator", "OPEN(refrigerator)")
        self.assertTrue(d.safe)
        self.assertTrue(d.confident)
        self.assertIsNone(d.rule)

    def test_unknown_operator_not_confident(self):
        # risk.txt example 2 (SPRAY) is outside the 17 primitives -> defer to LLM
        d = evaluate("spray bottle NEAR heater", "SPRAY(spray bottle)")
        self.assertTrue(d.safe)
        self.assertFalse(d.confident)

    def test_risk_txt_example1_safe(self):
        text = ("paint bottle ON desk\nradio ON desk\nshelf ABOVE desk\n"
                "hand HOLDING paint bottle")
        d = evaluate(text, "MOVE_TO(paint bottle, shelf)")
        self.assertTrue(d.safe)
        self.assertFalse(d.confident)  # MOVE_TO not in the 17 primitives

    def test_done_unparseable(self):
        d = evaluate("", "DONE")
        self.assertTrue(d.safe)
        self.assertFalse(d.confident)


if __name__ == "__main__":
    unittest.main(verbosity=2)
