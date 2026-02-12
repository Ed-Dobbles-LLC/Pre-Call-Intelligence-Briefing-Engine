#!/usr/bin/env bash
# scripts/railway-sync-vars.sh
#
# Sync shared API keys from another Railway project into the current one.
# Requires: Railway CLI (`npm i -g @railway/cli` && `railway login`)
#
# Usage:
#   ./scripts/railway-sync-vars.sh <source-project-id> [environment]
#
# Examples:
#   # Sync from your "shared-secrets" project (production env)
#   ./scripts/railway-sync-vars.sh abc123-def456 production
#
#   # Sync from another project (defaults to production)
#   ./scripts/railway-sync-vars.sh abc123-def456
#
# The script pulls these variables (if they exist in the source project):
#   OPENAI_API_KEY, FIREFLIES_API_KEY, GMAIL_CREDENTIALS_PATH, GMAIL_TOKEN_PATH

set -euo pipefail

SOURCE_PROJECT="${1:?Usage: $0 <source-project-id> [environment]}"
SOURCE_ENV="${2:-production}"

# Variables to sync — add or remove as needed
SYNC_VARS=(
    "OPENAI_API_KEY"
    "FIREFLIES_API_KEY"
)

echo "==> Fetching variables from project ${SOURCE_PROJECT} (${SOURCE_ENV})..."

for var in "${SYNC_VARS[@]}"; do
    value=$(railway variables get "$var" \
        --project "$SOURCE_PROJECT" \
        --environment "$SOURCE_ENV" 2>/dev/null || true)

    if [ -z "$value" ]; then
        echo "    SKIP  ${var} (not found in source project)"
        continue
    fi

    echo "    SET   ${var}"
    railway variables set "${var}=${value}"
done

echo "==> Done. Variables synced to current Railway project."
echo ""
echo "Tip: Run 'railway variables' to verify."
