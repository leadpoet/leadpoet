import os
import sys

# Import-time gateway clients require structurally valid values. Tests replace
# these clients before I/O and explicitly remove the variables when exercising
# missing-credential behavior, so no production credential is needed in CI.
os.environ.setdefault("SUPABASE_URL", "https://test.invalid")
os.environ.setdefault(
    "SUPABASE_SERVICE_ROLE_KEY",
    "test-only-service-role-placeholder",
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
