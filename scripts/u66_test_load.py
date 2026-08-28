#!/usr/bin/env python3
import time, sys

print("A: start", flush=True)
t = time.time()
import ml.m1_shared as m
print(f"B: import {time.time()-t:.3f}s", flush=True)

rp = str(__import__('pathlib').Path('repos/rust').resolve())
print(f"C: rp={rp}", flush=True)
print(f"D: _identity_loaded_for={m._identity_loaded_for!r}", flush=True)
print(f"E: equal={rp == m._identity_loaded_for}", flush=True)

print("F: calling build_identity_map", flush=True)
t2 = time.time()
from policy.identity import build_identity_map
print(f"G: imported in {time.time()-t2:.3f}s", flush=True)

t3 = time.time()
result = build_identity_map("repos/rust")
print(f"H: done {time.time()-t3:.3f}s, {len(result)} entries", flush=True)
