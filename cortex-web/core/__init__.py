"""Cortex hosted half: upload, ingest and search for Friday's document memory.

Small modules, imported by api/index.py. Heavy parsers (PyMuPDF, python-docx) are imported
lazily inside core.parse so the non-ingest endpoints start fast.
"""
