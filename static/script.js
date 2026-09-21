// Handle form submission
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Get form values
    const rbc = parseFloat(document.getElementById('rbc').value);
    const wbc = parseFloat(document.getElementById('wbc').value);
    const platelets = parseFloat(document.getElementById('platelets').value);
    const anc = document.getElementById('anc').value ? parseFloat(document.getElementById('anc').value) : null;
    const neutrophils_percent = document.getElementById('neutrophils_percent').value ? 
                                 parseFloat(document.getElementById('neutrophils_percent').value) : null;
    
    // Validate inputs
    if (!rbc || !wbc || !platelets) {
        showError('Please fill in all required fields (RBC, WBC, Platelets)');
        return;
    }
    
    // Show loading state
    const submitBtn = document.getElementById('submitBtn');
    const btnText = submitBtn.querySelector('.btn-text');
    const btnLoader = submitBtn.querySelector('.btn-loader');
    
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline-block';
    submitBtn.disabled = true;
    
    // Hide previous results/errors
    document.getElementById('results').style.display = 'none';
    document.getElementById('error').style.display = 'none';
    
    try {
        // Make API call
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                rbc: rbc,
                wbc: wbc,
                platelets: platelets,
                anc: anc,
                neutrophils_percent: neutrophils_percent
            })
        });
        
        let data;
        try {
            data = await response.json();
        } catch (jsonError) {
            throw new Error('Invalid response from server. Please try again.');
        }
        
        if (!response.ok) {
            const errorMsg = data.error || data.message || 'Prediction failed';
            throw new Error(typeof errorMsg === 'string' ? errorMsg : 'Prediction failed');
        }
        
        // Validate response structure
        if (!data || !data.predictions) {
            throw new Error('Invalid response format from server');
        }
        
        // Display results
        displayResults(data.predictions);
        
    } catch (error) {
        const errorMessage = error.message || String(error) || 'An error occurred while making predictions';
        showError(errorMessage);
        console.error('Prediction error:', error);
    } finally {
        // Reset button state
        btnText.style.display = 'inline-block';
        btnLoader.style.display = 'none';
        submitBtn.disabled = false;
    }
});

function displayResults(predictions) {
    // Validate predictions object
    if (!predictions || typeof predictions !== 'object') {
        showError('Invalid predictions data received from server');
        return;
    }
    
    // Display Anemia results
    if (predictions.anemia) {
        displayRiskResult('anemia', predictions.anemia);
    } else {
        displayRiskError('anemia', 'Anemia prediction not available');
    }
    
    // Display Neutropenia results
    if (predictions.neutropenia) {
        displayRiskResult('neutropenia', predictions.neutropenia);
    } else {
        displayRiskError('neutropenia', 'Neutropenia prediction not available');
    }
    
    // Display Thrombocytopenia results
    if (predictions.thrombocytopenia) {
        displayRiskResult('thrombocytopenia', predictions.thrombocytopenia);
    } else {
        displayRiskError('thrombocytopenia', 'Thrombocytopenia prediction not available');
    }
    
    // Show results container
    document.getElementById('results').style.display = 'block';
    
    // Scroll to results
    document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displayRiskResult(riskType, prediction) {
    // Validate prediction object
    if (!prediction || typeof prediction !== 'object') {
        displayRiskError(riskType, 'Invalid prediction data');
        return;
    }
    
    const badge = document.getElementById(`${riskType}Badge`);
    const probabilitiesDiv = document.getElementById(`${riskType}Probabilities`);
    
    if (!badge || !probabilitiesDiv) {
        console.error(`Elements not found for ${riskType}: badge or probabilities div`);
        return;
    }
    
    // Set risk badge
    const risk = prediction.risk || 'Unknown';
    badge.textContent = risk;
    badge.className = 'risk-badge ' + (risk ? risk.toLowerCase().replace(/\s+/g, '-') : 'unknown');
    
    // Display probabilities if available
    if (prediction.probabilities) {
        probabilitiesDiv.innerHTML = '';
        
        // Sort probabilities by value (descending)
        const sortedProbs = Object.entries(prediction.probabilities)
            .sort((a, b) => b[1] - a[1]);
        
        sortedProbs.forEach(([riskLevel, prob]) => {
            const probItem = document.createElement('div');
            probItem.className = 'prob-item';
            
            const probLabel = document.createElement('span');
            probLabel.className = 'prob-label';
            probLabel.textContent = riskLevel;
            
            const probBarContainer = document.createElement('div');
            probBarContainer.className = 'prob-bar-container';
            
            const probBar = document.createElement('div');
            probBar.className = `prob-bar ${riskLevel.toLowerCase()}`;
            probBar.style.width = `${prob * 100}%`;
            probBar.textContent = prob >= 0.01 ? `${(prob * 100).toFixed(1)}%` : '';
            
            const probValue = document.createElement('span');
            probValue.className = 'prob-value';
            probValue.textContent = `${(prob * 100).toFixed(1)}%`;
            
            probBarContainer.appendChild(probBar);
            probItem.appendChild(probLabel);
            probItem.appendChild(probBarContainer);
            probItem.appendChild(probValue);
            
            probabilitiesDiv.appendChild(probItem);
        });
    } else if (prediction.error) {
        probabilitiesDiv.innerHTML = `<p style="color: #e74c3c; font-style: italic;">${prediction.error}</p>`;
    } else {
        probabilitiesDiv.innerHTML = '<p style="color: #7f8c8d;">Probabilities not available</p>';
    }
}

function displayRiskError(riskType, errorMessage) {
    const badge = document.getElementById(`${riskType}Badge`);
    const probabilitiesDiv = document.getElementById(`${riskType}Probabilities`);
    
    if (badge) {
        badge.textContent = 'Error';
        badge.className = 'risk-badge error';
    }
    
    if (probabilitiesDiv) {
        probabilitiesDiv.innerHTML = `<p style="color: #e74c3c; font-style: italic;">${errorMessage}</p>`;
    }
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    const errorText = document.getElementById('errorText');
    
    if (!errorDiv || !errorText) {
        console.error('Error display elements not found');
        alert('Error: ' + (message || 'An unknown error occurred'));
        return;
    }
    
    // Ensure message is a string
    const errorMessage = typeof message === 'string' ? message : String(message || 'An unknown error occurred');
    errorText.textContent = errorMessage;
    errorDiv.style.display = 'block';
    
    // Scroll to error
    errorDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function resetForm() {
    // Reset form
    document.getElementById('predictionForm').reset();
    
    // Hide results and errors
    document.getElementById('results').style.display = 'none';
    document.getElementById('error').style.display = 'none';
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Auto-calculate ANC if neutrophils percentage is provided
document.getElementById('neutrophils_percent').addEventListener('input', function() {
    const wbc = parseFloat(document.getElementById('wbc').value);
    const neutrophils = parseFloat(this.value);
    const ancInput = document.getElementById('anc');
    
    if (wbc && neutrophils && !ancInput.value) {
        // Calculate ANC (assuming WBC is in thousands/μL)
        const wbcCells = wbc > 1000 ? wbc : wbc * 1000;
        const anc = (wbcCells * neutrophils / 100) / 1000; // Convert to thousands/μL
        
        // Show calculated ANC (optional - you can auto-fill or just show a hint)
        // ancInput.value = anc.toFixed(2);
    }
});


