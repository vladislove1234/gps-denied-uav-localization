"""The segmentation class set, importable without the training stack (torch, smp)."""

CLASSES = ("background", "building", "track", "woody", "field", "water")
BACKGROUND = CLASSES.index("background")
