"""SEM subsystem package.

This package is introduced first as a low-risk structural landing zone.
During migration, root-level legacy paths remain available as compatibility shims.

Keep this package initializer intentionally light to avoid import cycles while
legacy root-level modules still re-export moved implementations.
"""
