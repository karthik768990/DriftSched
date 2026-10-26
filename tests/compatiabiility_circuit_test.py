from quantum_drift_scheduler.data.compatibility import load_compatibility_registry

registry = load_compatibility_registry()
print("Compatible pairs:", len(registry.compatible_pairs()))
print(registry.get_status("Gx(Gw)Gy", "Gx(Gw)^2Gy"))
print(registry.is_compatible("Gx(Gw)^8192Gy", "Gx(Gw)^4096Gy"))
