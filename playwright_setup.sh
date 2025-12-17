#!/bin/bash
# Railway Playwright Setup Script
# This installs Chromium browser for Playwright

echo "Installing Playwright browsers..."
playwright install chromium

echo "Installing system dependencies for Playwright..."
playwright install-deps chromium || true

echo "Playwright setup complete!"
