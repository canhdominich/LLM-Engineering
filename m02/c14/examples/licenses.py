import httpx

OLLAMA = "http://localhost:11434"
sizes = {m["name"]: m["size"] for m in httpx.get(f"{OLLAMA}/api/tags").json()["models"]}
CLAUSES = ["Hosted Service", "Prohibited Use Policy", "restrict (remotely", "must make reasonable efforts to use the latest"]

for name in ("qwen3:1.7b", "gemma3:270m"):
    d = httpx.post(f"{OLLAMA}/api/show", json={"model": name}).json()
    lic = d.get("license", "")
    info = d["model_info"]
    params = info["general.parameter_count"] / 1e9
    ctx = next(v for k, v in info.items() if k.endswith(".context_length"))
    print(f"{name}: {params:.2f}B params, {d['details']['quantization_level']}, {sizes[name] / 1e6:.0f} MB, "
          f"context {ctx}, capabilities {d['capabilities']}")
    print(f"  license: {lic.strip().splitlines()[0].strip()!r}, {len(lic):,} chars")
    found = [c for c in CLAUSES if c in lic]
    print(f"  clauses to read before shipping: {found or 'none of the listed ones'}")
