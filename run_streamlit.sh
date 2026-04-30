#!/usr/bin/env bash
# Run the Streamlit app with the project venv Python (avoids global /usr/local/python).
set -e
cd "$(dirname "$0")"
if [[ ! -x "venv/bin/python" ]]; then
  echo "Error: venv/bin/python not found. Create the venv first:"
  echo "  python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi
exec "./venv/bin/python" -m streamlit run app.py "$@"
