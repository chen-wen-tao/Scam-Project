#!/bin/bash
# Script to find optimal threshold with new multi_category prompt

echo "=========================================="
echo "Finding Optimal Threshold"
echo "Using NEW multi_category prompt"
echo "=========================================="
echo ""

# Test multiple thresholds
python test_thresholds.py \
    --input data/cfpb-complaints-2025-11-03-12-03.csv \
    --thresholds 50 60 70 80 90 \
    --workers 2

echo ""
echo "=========================================="
echo "Threshold testing complete!"
echo "Check the output above for the best threshold"
echo "=========================================="

