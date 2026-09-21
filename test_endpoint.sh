#!/bin/bash

# Test script for /predict endpoint
# Make sure your Flask server is running on http://localhost:5000

echo "Testing /predict endpoint..."
echo ""

# Test 1: Basic request (without ANC/neutrophils)
echo "Test 1: Basic request (RBC, WBC, Platelets only)"
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "rbc": 4.5,
    "wbc": 7.0,
    "platelets": 250
}'
echo ""
echo ""

# Test 2: Full request with ANC
echo "Test 2: Full request with ANC"
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "rbc": 4.5,
    "wbc": 7.0,
    "platelets": 250,
    "anc": 3.5
}'
echo ""
echo ""

# Test 3: Request with neutrophils_percent
echo "Test 3: Request with neutrophils_percent"
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "rbc": 3.8,
    "wbc": 5.5,
    "platelets": 180,
    "neutrophils_percent": 45
}'
echo ""
echo ""

# Test 4: Health check
echo "Test 4: Health check endpoint"
curl -X GET http://localhost:5000/health
echo ""
echo ""

echo "All tests completed!"



