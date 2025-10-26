#!/bin/bash
# Download SOT batch data
# This is a template script - actual implementation depends on where SOT data is hosted
#
# Usage:
#   bash scripts/download_batch.sh --start-date 2025-01-01 --end-date 2025-01-31
#   bash scripts/download_batch.sh --start-date 2025-01-01 --end-date 2025-01-31 --output-dir ./custom/path

set -e  # Exit on error
set -u  # Exit on undefined variable

# Default values
OUTPUT_DIR="./data"
START_DATE=""
END_DATE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --start-date)
      START_DATE="$2"
      shift 2
      ;;
    --end-date)
      END_DATE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 --start-date YYYY-MM-DD --end-date YYYY-MM-DD [--output-dir DIR]"
      echo ""
      echo "Options:"
      echo "  --start-date    Start date in YYYY-MM-DD format (required)"
      echo "  --end-date      End date in YYYY-MM-DD format (required)"
      echo "  --output-dir    Output directory (default: ./data)"
      echo "  -h, --help      Show this help message"
      echo ""
      echo "Example:"
      echo "  $0 --start-date 2025-01-01 --end-date 2025-01-31"
      exit 0
      ;;
    *)
      echo "Error: Unknown option: $1"
      echo "Run '$0 --help' for usage information"
      exit 1
      ;;
  esac
done

# Validate required inputs
if [ -z "$START_DATE" ] || [ -z "$END_DATE" ]; then
  echo "Error: --start-date and --end-date are required"
  echo "Run '$0 --help' for usage information"
  exit 1
fi

# Validate date format (basic check)
if ! [[ "$START_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "Error: Invalid start date format. Expected YYYY-MM-DD, got: $START_DATE"
  exit 1
fi

if ! [[ "$END_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "Error: Invalid end date format. Expected YYYY-MM-DD, got: $END_DATE"
  exit 1
fi

# Create output directory
echo "Creating output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Display download parameters
echo "========================================="
echo "SOT Batch Data Download"
echo "========================================="
echo "Start Date:  $START_DATE"
echo "End Date:    $END_DATE"
echo "Output Dir:  $OUTPUT_DIR"
echo "========================================="

# Template implementation
# In production, replace this section with actual download logic
echo ""
echo "NOTE: This is a template script."
echo "Real implementation would download from SOT API or S3 bucket."
echo ""
echo "Expected files to download:"
echo "  - alerts.parquet      (Alert data with features)"
echo "  - features.parquet    (Additional feature data)"
echo "  - clusters.parquet    (Cluster membership data)"
echo ""

# Example of what the actual implementation might look like:
# 
# # Option 1: Download from S3
# aws s3 cp "s3://sot-data/batches/${START_DATE}_${END_DATE}/alerts.parquet" \
#           "$OUTPUT_DIR/alerts.parquet"
# aws s3 cp "s3://sot-data/batches/${START_DATE}_${END_DATE}/features.parquet" \
#           "$OUTPUT_DIR/features.parquet"
# aws s3 cp "s3://sot-data/batches/${START_DATE}_${END_DATE}/clusters.parquet" \
#           "$OUTPUT_DIR/clusters.parquet"
#
# # Option 2: Download from HTTP API
# curl -o "$OUTPUT_DIR/alerts.parquet" \
#      "https://sot-api.example.com/batches/alerts?start=$START_DATE&end=$END_DATE"
# curl -o "$OUTPUT_DIR/features.parquet" \
#      "https://sot-api.example.com/batches/features?start=$START_DATE&end=$END_DATE"
# curl -o "$OUTPUT_DIR/clusters.parquet" \
#      "https://sot-api.example.com/batches/clusters?start=$START_DATE&end=$END_DATE"

# Create placeholder files for template purposes
echo "Creating placeholder files for template demonstration..."
touch "$OUTPUT_DIR/alerts.parquet"
touch "$OUTPUT_DIR/features.parquet"
touch "$OUTPUT_DIR/clusters.parquet"

# Validate downloaded files
echo ""
echo "Validating downloaded files..."

validate_file() {
  local file=$1
  if [ ! -f "$file" ]; then
    echo "Error: Expected file not found: $file"
    exit 1
  fi
  echo "  ✓ $file exists"
}

validate_file "$OUTPUT_DIR/alerts.parquet"
validate_file "$OUTPUT_DIR/features.parquet"
validate_file "$OUTPUT_DIR/clusters.parquet"

echo ""
echo "========================================="
echo "Download completed successfully!"
echo "Files saved to: $OUTPUT_DIR"
echo "========================================="

exit 0