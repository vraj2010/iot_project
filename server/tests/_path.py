"""Lets the tests import the server's modules (they live one folder up)."""

import os
import pathlib
import sys

# Keep the tests away from the real Cortex database that this laptop's .env may point at: an empty value
# here wins over .env (config only fills in what isn't set). Cortex tests switch it on with a fake.
os.environ["CORTEX_DATABASE_URL"] = ""

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
