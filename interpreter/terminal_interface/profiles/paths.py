"""Where profiles live, and the profile-format version.

Separate from profiles.py so the migration code can share these without
importing back into the module that imports it.
"""

import glob
import os

from ..utils.oi_dir import oi_dir

profile_dir = os.path.join(oi_dir, "profiles")
user_default_profile_path = os.path.join(profile_dir, "default.yaml")

here = os.path.abspath(os.path.dirname(__file__))
oi_default_profiles_path = os.path.join(here, "defaults")
default_profiles_paths = glob.glob(os.path.join(oi_default_profiles_path, "*"))
default_profiles_names = [os.path.basename(path) for path in default_profiles_paths]

# Constant to hold the version number
OI_VERSION = "0.2.5"
