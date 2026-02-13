#!/usr/bin/env python3
"""One-time Gmail OAuth setup script.

Walks you through getting the 3 env vars needed for Railway:
  GOOGLE_CLIENT_ID
  GOOGLE_CLIENT_SECRET
  GOOGLE_REFRESH_TOKEN

Usage:
  1. Go to https://console.cloud.google.com/apis/credentials
  2. Create an OAuth 2.0 Client ID (type: Desktop app)
  3. Download the JSON → save as credentials.json in this repo root
  4. Run: python scripts/setup_gmail_oauth.py
  5. It opens a browser, you log in, approve read-only Gmail access
  6. It prints the 3 env vars → paste them into Railway

The app only requests gmail.readonly scope (can read, never send/delete).
"""

import json
import sys
from pathlib import Path

def main():
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("Install dependencies first:")
        print("  pip install google-auth google-auth-oauthlib google-api-python-client")
        sys.exit(1)

    creds_path = Path("credentials.json")
    if not creds_path.exists():
        print("""
=== Step 1: Create Google Cloud OAuth Credentials ===

1. Go to: https://console.cloud.google.com/apis/credentials
2. Click "+ CREATE CREDENTIALS" → "OAuth client ID"
3. Application type: "Desktop app"
4. Name it anything (e.g. "Pre-Call Intelligence")
5. Click "Download JSON" on the created credential
6. Save the file as: credentials.json  (in this repo's root directory)

Then re-run this script.
""")
        sys.exit(1)

    # Also need Gmail API enabled
    with open(creds_path) as f:
        creds_data = json.load(f)

    # Extract client info
    installed = creds_data.get("installed") or creds_data.get("web", {})
    client_id = installed.get("client_id", "")
    client_secret = installed.get("client_secret", "")

    if not client_id:
        print("ERROR: credentials.json doesn't contain client_id. Re-download from Google Cloud Console.")
        sys.exit(1)

    print("""
=== Step 2: Authorize Gmail Access ===

A browser window will open. Log into the Google account whose
Gmail you want to sync, and approve read-only access.

(If you see "This app isn't verified", click "Advanced" → "Go to ... (unsafe)")
""")

    SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

    flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
    creds = flow.run_local_server(port=8090, prompt="consent", access_type="offline")

    if not creds.refresh_token:
        print("ERROR: No refresh token received. Try deleting token.json and re-running.")
        sys.exit(1)

    print(f"""
=== Done! Add these 3 env vars to Railway ===

GOOGLE_CLIENT_ID={client_id}
GOOGLE_CLIENT_SECRET={client_secret}
GOOGLE_REFRESH_TOKEN={creds.refresh_token}

Go to: Railway Dashboard → your project → Variables
Paste each one, then redeploy.

After deploy, click "Sync Now" — emails will start flowing in.
""")


if __name__ == "__main__":
    main()
