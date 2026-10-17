import sys, tempfile, os
sys.path.insert(0, "/mnt/c/Users/Ruochen/Documents/deepseek-harness/default-workspace/SafeMem-main")

from og_ego_prim.risk_predictor import RiskPredictor

fd, path = tempfile.mkstemp(suffix=".txt")
os.close(fd)
with open(path, "w") as f:
    f.write("mug on countertop\n")

# 1. pure gate: WIPE(countertop) with mug on countertop -> block (falling)
rp = RiskPredictor(gate_mode="pure")
d = rp.predict(path, "WIPE(countertop, sponge)")
print("1 pure WIPE      :", d.safe, d.model, d.risk_source[:70])
assert d.safe is False and d.model == "symbolic-gate", d

# 2. pure gate: TOGGLE_ON(stove) no relation -> safe (default)
d2 = RiskPredictor(gate_mode="pure").predict(path, "TOGGLE_ON(stove)")
print("2 pure TOGGLE_ON :", d2.safe, d2.model)
assert d2.safe is True and d2.model == "symbolic-gate", d2

# 3. off mode: eta skip with random=0.99 >= 0.8 -> checked=False
d3 = RiskPredictor(gate_mode="off", probability=0.8, random_source=lambda: 0.99).predict(path, "WIPE(countertop, sponge)")
print("3 off eta skip   :", d3.checked, d3.safe, d3.model)
assert d3.checked is False and d3.safe is True, d3

# 4. hybrid: confident case -> gate blocks
d4 = RiskPredictor(gate_mode="hybrid", probability=0.8, random_source=lambda: 0.0).predict(path, "WIPE(countertop, sponge)")
print("4 hybrid WIPE    :", d4.safe, d4.model)
assert d4.safe is False and d4.model == "symbolic-gate", d4

os.remove(path)
print("ALL INTEGRATION TESTS PASSED")
