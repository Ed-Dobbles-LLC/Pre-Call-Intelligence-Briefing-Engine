#!/usr/bin/env bash
# scripts/railway-bootstrap.sh
#
# One-time setup: run this after `railway link` on ANY new project.
# It reads your shared keys from ~/.railway-shared-keys and sets them.
#
# First-time setup (do this ONCE):
#   1. Install Railway CLI:  npm i -g @railway/cli
#   2. Log in:               railway login
#   3. Create your key file: cp scripts/railway-shared-keys.example ~/.railway-shared-keys
#   4. Fill in your real keys in ~/.railway-shared-keys
#
# For each new project:
#   1. cd into the project
#   2. railway link            (pick the Railway project)
#   3. ./scripts/railway-bootstrap.sh
#
# That's it. Works for this project, the next one, and every one after.

set -euo pipefail

KEYS_FILE="$HOME/.railway-shared-keys"

if [ ! -f "$KEYS_FILE" ]; then
    echo "ERROR: $KEYS_FILE not found."
    echo ""
    echo "Create it once with your shared API keys:"
    echo "  cp scripts/railway-shared-keys.example ~/.railway-shared-keys"
    echo "  nano ~/.railway-shared-keys   # fill in your real values"
    exit 1
fi

echo "==> Reading shared keys from $KEYS_FILE"

while IFS='=' read -r key value; do
    # Skip blank lines and comments
    [[ -z "$key" || "$key" =~ ^# ]] && continue
    # Trim whitespace
    key=$(echo "$key" | xargs)
    value=$(echo "$value" | xargs)

    echo "    SET  $key"
    railway variables set "${key}=${value}"
done < "$KEYS_FILE"

echo ""
echo "==> Done. Run 'railway variables' to verify."
