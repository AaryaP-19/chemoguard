# Test script for /predict endpoint (PowerShell)
# Make sure your Flask server is running on http://localhost:5000

Write-Host "Testing /predict endpoint..." -ForegroundColor Green
Write-Host ""

# Test 1: Basic request (without ANC/neutrophils)
Write-Host "Test 1: Basic request (RBC, WBC, Platelets only)" -ForegroundColor Yellow
$body1 = @{
    rbc = 4.5
    wbc = 7.0
    platelets = 250
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method POST -Body $body1 -ContentType "application/json" | ConvertTo-Json -Depth 10
Write-Host ""

# Test 2: Full request with ANC
Write-Host "Test 2: Full request with ANC" -ForegroundColor Yellow
$body2 = @{
    rbc = 4.5
    wbc = 7.0
    platelets = 250
    anc = 3.5
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method POST -Body $body2 -ContentType "application/json" | ConvertTo-Json -Depth 10
Write-Host ""

# Test 3: Request with neutrophils_percent
Write-Host "Test 3: Request with neutrophils_percent" -ForegroundColor Yellow
$body3 = @{
    rbc = 3.8
    wbc = 5.5
    platelets = 180
    neutrophils_percent = 45
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method POST -Body $body3 -ContentType "application/json" | ConvertTo-Json -Depth 10
Write-Host ""

# Test 4: Health check
Write-Host "Test 4: Health check endpoint" -ForegroundColor Yellow
Invoke-RestMethod -Uri "http://localhost:5000/health" -Method GET | ConvertTo-Json
Write-Host ""

Write-Host "All tests completed!" -ForegroundColor Green



