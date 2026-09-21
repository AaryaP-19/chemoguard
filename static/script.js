// CSRF token for mutating requests. Flask-WTF checks the X-CSRFToken
// header; the server renders the value into a meta tag in index.html.
function csrfToken() {
    const el = document.querySelector('meta[name="csrf-token"]');
    return el ? el.getAttribute('content') : '';
}

// Global state
let currentUser = null;
let currentPredictionId = null;
let extractedData = null;
let unusableWithoutUnits = {};

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    checkAuth();
    setupEventListeners();
    loadHistory();
    loadReports();
});

// Check authentication status
async function checkAuth() {
    try {
        const response = await fetch('/api/auth/check_auth', {
            credentials: 'include'
        });
        const data = await response.json();
        
        if (data.authenticated) {
            currentUser = data.user;
            showAuthenticatedUI();
        } else {
            showUnauthenticatedUI();
        }
    } catch (error) {
        console.error('Auth check error:', error);
        showUnauthenticatedUI();
    }
}

// Show authenticated UI
function showAuthenticatedUI() {
    document.getElementById('navAuth').style.display = 'none';
    document.getElementById('navUser').style.display = 'flex';
    document.getElementById('userName').textContent = currentUser.username;
    document.getElementById('uploadLink').style.display = 'block';
    document.getElementById('historyLink').style.display = 'block';
    document.getElementById('profileLink').style.display = 'block';
}

// Show unauthenticated UI
function showUnauthenticatedUI() {
    document.getElementById('navAuth').style.display = 'flex';
    document.getElementById('navUser').style.display = 'none';
    document.getElementById('uploadLink').style.display = 'none';
    document.getElementById('historyLink').style.display = 'none';
    document.getElementById('profileLink').style.display = 'none';
}

// Setup event listeners
function setupEventListeners() {
    // Navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const section = e.target.getAttribute('data-section');
            showSection(section);
        });
    });

    // Auth buttons
    document.getElementById('loginBtn').addEventListener('click', () => showModal('loginModal'));
    document.getElementById('registerBtn').addEventListener('click', () => showModal('registerModal'));
    document.getElementById('logoutBtn').addEventListener('click', logout);
    document.getElementById('closeLogin').addEventListener('click', () => hideModal('loginModal'));
    document.getElementById('closeRegister').addEventListener('click', () => hideModal('registerModal'));

    // Forms
    document.getElementById('loginForm').addEventListener('submit', handleLogin);
    document.getElementById('registerForm').addEventListener('submit', handleRegister);
    document.getElementById('predictionForm').addEventListener('submit', handlePrediction);
    document.getElementById('profileForm').addEventListener('submit', handleProfileUpdate);

    // File upload
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    const chooseFileBtn = document.getElementById('chooseFileBtn');
    
    fileInput.addEventListener('change', handleFileSelect);
    
    // Button click handler - stop propagation to prevent double trigger
    if (chooseFileBtn) {
        chooseFileBtn.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent triggering uploadArea click
            fileInput.click();
        });
    }
    
    // Upload area click handler - but not if clicking the button
    uploadArea.addEventListener('click', (e) => {
        // Only trigger if not clicking the button or its children
        if (e.target !== chooseFileBtn && !chooseFileBtn.contains(e.target)) {
            fileInput.click();
        }
    });
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            // Create a new FileList-like object and trigger change
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(files[0]);
            fileInput.files = dataTransfer.files;
            // Trigger change event manually
            const changeEvent = new Event('change', { bubbles: true });
            fileInput.dispatchEvent(changeEvent);
        }
    });
}

// Show section
function showSection(sectionName) {
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    
    document.getElementById(sectionName + 'Section').classList.add('active');
    const navLink = document.querySelector(`[data-section="${sectionName}"]`);
    if (navLink) {
        navLink.classList.add('active');
    }
    
    // Load section-specific data
    if (sectionName === 'profile' && currentUser) {
        loadProfile();
    } else if (sectionName === 'history' && currentUser) {
        loadHistory();
    } else if (sectionName === 'upload' && currentUser) {
        loadReports();
    }
}

// Modal functions
function showModal(modalId) {
    document.getElementById(modalId).classList.add('active');
}

function hideModal(modalId) {
    document.getElementById(modalId).classList.remove('active');
}

// Login
async function handleLogin(e) {
    e.preventDefault();
    const username = document.getElementById('loginUsername').value;
    const password = document.getElementById('loginPassword').value;

    try {
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrfToken() },
            credentials: 'include',
            body: JSON.stringify({ username, password })
        });

        const data = await response.json();
        
        if (response.ok) {
            currentUser = data.user;
            showAuthenticatedUI();
            hideModal('loginModal');
            showError('Login successful!', 'success');
            loadHistory();
            loadReports();
        } else {
            showError(data.error || 'Login failed');
        }
    } catch (error) {
        showError('Network error. Please try again.');
    }
}

// Register
async function handleRegister(e) {
    e.preventDefault();
    const username = document.getElementById('regUsername').value;
    const email = document.getElementById('regEmail').value;
    const fullName = document.getElementById('regFullName').value;
    const password = document.getElementById('regPassword').value;

    try {
        const response = await fetch('/api/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrfToken() },
            credentials: 'include',
            body: JSON.stringify({ username, email, full_name: fullName, password })
        });

        const data = await response.json();
        
        if (response.ok) {
            currentUser = data.user;
            showAuthenticatedUI();
            hideModal('registerModal');
            showError('Registration successful!', 'success');
        } else {
            showError(data.error || 'Registration failed');
        }
    } catch (error) {
        showError('Network error. Please try again.');
    }
}

// Logout
async function logout() {
    try {
        await fetch('/api/auth/logout', {
            method: 'POST',
            headers: { 'X-CSRFToken': csrfToken() },
            credentials: 'include'
        });
        currentUser = null;
        showUnauthenticatedUI();
        showSection('predict');
        showError('Logged out successfully', 'success');
    } catch (error) {
        console.error('Logout error:', error);
    }
}

// Trigger prediction (can be called manually or from form submit)
// Make it globally accessible to avoid scope issues
// Read a value together with the unit selected beside it.
// Returns null when the field is blank, so an optional analyte can be
// omitted -- but never a number without the unit that goes with it.
function readQuantity(id, fixedUnit) {
    const input = document.getElementById(id);
    if (!input || input.value === '') return null;

    const value = parseFloat(input.value);
    if (isNaN(value)) return null;

    const select = document.getElementById(id + '_unit');
    const unit = fixedUnit || (select ? select.value : null);
    if (!unit) return null;

    return { value: value, unit: unit };
}

window.triggerPrediction = async function triggerPrediction() {
    // Each entry is {value, unit}. The server refuses a bare number with
    // 422, so there is nothing to gain by sending one.
    const payload = {
        rbc: readQuantity('rbc'),
        wbc: readQuantity('wbc'),
        platelets: readQuantity('platelets'),
        anc: readQuantity('anc'),
        neutrophils_percent: readQuantity('neutrophils_percent', '%')
    };

    console.log('Triggering prediction with values:', payload);

    if (!payload.rbc || !payload.wbc || !payload.platelets) {
        showError('Please fill in all required fields (RBC, WBC, Platelets) with valid numbers and select a unit for each');
        return;
    }

    const submitBtn = document.getElementById('submitBtn');
    const btnText = submitBtn ? submitBtn.querySelector('.btn-text') : null;
    const btnLoader = submitBtn ? submitBtn.querySelector('.btn-loader') : null;
    
    if (submitBtn && btnText && btnLoader) {
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline-block';
        submitBtn.disabled = true;
    }

    // Show loading message
    showError('Processing prediction...', 'info');

    try {
        const endpoint = currentUser ? '/api/predict' : '/predict';
        console.log('Sending prediction request to:', endpoint);
        
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrfToken() },
            credentials: 'include',
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        console.log('Prediction response:', data);
        
        if (response.ok) {
            if (data.prediction_id) {
                currentPredictionId = data.prediction_id;
                const resultActions = document.getElementById('resultActions');
                if (resultActions) {
                    resultActions.style.display = 'flex';
                }
            }
            // Use combined_toxicity or combined_severity
            const combinedToxicity = data.combined_toxicity || data.combined_severity || 'Low';
            displayResults(data.predictions, combinedToxicity);
            
            // Scroll to results after a short delay
            setTimeout(() => {
                const resultsDiv = document.getElementById('results');
                if (resultsDiv) {
                    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }, 200);
            
            showError('Prediction completed successfully!', 'success');
            
            if (currentUser) {
                loadHistory();
            }
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (error) {
        showError('Network error. Please try again.');
        console.error('Prediction error:', error);
    } finally {
        if (submitBtn && btnText && btnLoader) {
            btnText.style.display = 'inline-block';
            btnLoader.style.display = 'none';
            submitBtn.disabled = false;
        }
    }
};

// Prediction form handler
async function handlePrediction(e) {
    e.preventDefault();
    await window.triggerPrediction();
}

// Display results
function displayResults(predictions, combinedSeverity) {
    const resultsDiv = document.getElementById('results');
    const gridDiv = document.getElementById('resultsGrid');
    
    gridDiv.innerHTML = '';
    
    // Combined hematological toxicity
    if (combinedSeverity) {
        const severityDiv = document.getElementById('combinedSeverity');
        const severityText = combinedSeverity === 'Moderate' ? 'MODERATE' : combinedSeverity.toUpperCase();
        severityDiv.innerHTML = `
            <h3>Combined Hematological Toxicity</h3>
            <div class="toxicity-badge ${combinedSeverity.toLowerCase()}">${severityText}</div>
        `;
        severityDiv.className = `combined-severity ${combinedSeverity.toLowerCase()}`;
        severityDiv.style.display = 'block';
    }

    // Display each risk type
    ['anemia', 'neutropenia', 'thrombocytopenia'].forEach(riskType => {
        if (predictions[riskType]) {
            const card = createRiskCard(riskType, predictions[riskType]);
            gridDiv.appendChild(card);
        }
    });

    resultsDiv.style.display = 'block';
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

// Create risk card
function createRiskCard(riskType, prediction) {
    const card = document.createElement('div');
    card.className = 'result-card';
    
    const risk = prediction.risk || 'Unknown';
    const riskClass = risk.toLowerCase().replace(/\s+/g, '-');
    
    card.innerHTML = `
        <div class="result-header">
            <h3>${riskType.charAt(0).toUpperCase() + riskType.slice(1)} Risk</h3>
            <span class="risk-badge ${riskClass}">${risk}</span>
        </div>
        <div class="probability-bar">
            ${prediction.probabilities ? 
                Object.entries(prediction.probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .map(([level, prob]) => `
                        <div class="prob-item">
                            <span class="prob-label">${level}</span>
                            <div class="prob-bar-container">
                                <div class="prob-bar ${level.toLowerCase()}" style="width: ${prob * 100}%">
                                    ${prob >= 0.01 ? `${(prob * 100).toFixed(1)}%` : ''}
                                </div>
                            </div>
                            <span class="prob-value">${(prob * 100).toFixed(1)}%</span>
                        </div>
                    `).join('') : 
                '<p style="color: #7f8c8d;">Probabilities not available</p>'
            }
        </div>
    `;
    
    return card;
}

// File upload
async function handleFileSelect(e) {
    const fileInput = e.target;
    const file = fileInput.files[0];
    if (!file) return;

    // Prevent multiple simultaneous uploads
    if (fileInput.disabled) {
        return;
    }
    
    // Disable input during upload
    fileInput.disabled = true;

    const formData = new FormData();
    formData.append('file', file);

    document.getElementById('uploadProgress').style.display = 'block';
    document.getElementById('extractedData').style.display = 'none';
    document.getElementById('progressText').textContent = 'Processing with GPT-4o-mini...';

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            // No Content-Type here: the browser must set the multipart boundary.
            headers: { 'X-CSRFToken': csrfToken() },
            credentials: 'include',
            body: formData
        });

        const data = await response.json();
        
        if (response.ok) {
            extractedData = data.extracted_parameters || {};
            unusableWithoutUnits = data.unusable_without_units || {};
            console.log('Extracted data received:', extractedData);
            console.log('Unusable without units:', unusableWithoutUnits);
            displayExtractedData(extractedData, unusableWithoutUnits);
            showError(`File processed successfully! Confidence: ${data.confidence?.toFixed(1) || 'N/A'}%`, 'success');
            if (currentUser) {
                loadReports();
            }
        } else {
            showError(data.error || 'Upload failed');
            extractedData = null;
        }
    } catch (error) {
        showError('Network error. Please try again.');
        console.error('Upload error:', error);
    } finally {
        document.getElementById('uploadProgress').style.display = 'none';
        // Re-enable input and reset value to allow same file upload again
        fileInput.disabled = false;
        fileInput.value = ''; // Reset so same file can be selected again
    }
}

// Display extracted data.
// Each entry is {canonical: {value, unit}, as_reported: {value, unit},
// unit_source}. Values are shown as the report printed them, with the
// canonical form beside it, so the operator can check the transcription
// against the page rather than against a converted number.
function displayExtractedData(data, unusable) {
    const grid = document.getElementById('extractedGrid');
    grid.innerHTML = '';

    if (!data || Object.keys(data).length === 0) {
        grid.innerHTML = '<p style="color: #7f8c8d; text-align: center; padding: 20px;">No parameters extracted. Please try again with a clearer image.</p>';
        document.getElementById('extractedData').style.display = 'block';
        return;
    }
    
    const paramLabels = {
        'rbc': 'RBC (Red Blood Cells)',
        'wbc': 'WBC (White Blood Cells)',
        'platelets': 'Platelets',
        'anc': 'ANC (Absolute Neutrophil Count)',
        'neutrophils_percent': 'Neutrophils %',
        'hb': 'Hemoglobin (Hb)',
        'hematocrit': 'Hematocrit',
        'mcv': 'MCV',
        'mch': 'MCH',
        'mchc': 'MCHC'
    };
    
    Object.entries(data).forEach(([key, entry]) => {
        if (!entry || !entry.as_reported) return;
        const item = document.createElement('div');
        item.className = 'extracted-item';
        const label = paramLabels[key] || key.toUpperCase();
        const reported = `${entry.as_reported.value} ${entry.as_reported.unit}`;
        const canonical = `${entry.canonical.value} ${entry.canonical.unit}`;
        item.innerHTML = `
            <label>${label}</label>
            <value>${reported}</value>
            ${reported === canonical ? '' : `<small class="canonical">= ${canonical}</small>`}
        `;
        grid.appendChild(item);
    });

    // Rows the extractor read but could not attach a unit to. Shown rather
    // than dropped: the operator needs to tell "not on the report" from
    // "on the report, unit unreadable", because only the second is
    // something they can fix by typing the unit in.
    Object.entries(unusable || {}).forEach(([key, entry]) => {
        const item = document.createElement('div');
        item.className = 'extracted-item extracted-item-unusable';
        const label = paramLabels[key] || key.toUpperCase();
        item.innerHTML = `
            <label>${label}</label>
            <value>${entry.value} <em>(no unit read)</em></value>
            <small class="canonical">Enter this value and its unit below; it will not be used until you do.</small>
        `;
        grid.appendChild(item);
    });

    if (grid.children.length === 0) {
        grid.innerHTML = '<p style="color: #7f8c8d; text-align: center; padding: 20px;">No valid parameters found in the image.</p>';
    }

    document.getElementById('extractedData').style.display = 'block';
}

// Copy one extracted quantity into its input and unit selector.
// Prefers the unit the report printed; falls back to the canonical unit
// when the selector has no option for it (e.g. a report in 10^9/L).
function setQuantity(id, entry) {
    const input = document.getElementById(id);
    if (!input || !entry || !entry.canonical) return false;

    const select = document.getElementById(id + '_unit');
    const options = select ? Array.from(select.options).map(o => o.value) : [];
    const preferred = entry.as_reported && options.includes(entry.as_reported.unit)
        ? entry.as_reported
        : entry.canonical;

    input.value = preferred.value;
    if (select && options.includes(preferred.unit)) {
        select.value = preferred.unit;
    }
    return true;
}

// Use extracted data - make it globally accessible for onclick handler
window.useExtractedData = function() {
    console.log('useExtractedData called');
    console.log('Current extractedData:', extractedData);
    
    if (!extractedData || Object.keys(extractedData).length === 0) {
        showError('No extracted data available. Please upload a file first.');
        return;
    }
    
    let valuesSet = false;

    // Value and unit travel together, for every field.
    valuesSet = setQuantity('rbc', extractedData.rbc) || valuesSet;
    valuesSet = setQuantity('wbc', extractedData.wbc) || valuesSet;
    valuesSet = setQuantity('platelets', extractedData.platelets) || valuesSet;
    setQuantity('anc', extractedData.anc);
    setQuantity('neutrophils_percent', extractedData.neutrophils_percent);

    if (!valuesSet) {
        showError('No valid values (RBC, WBC, Platelets) found in extracted data. Please check the extracted parameters.');
        return;
    }
    
    // Switch to predict section
    showSection('predict');
    
    // Scroll to form and auto-trigger prediction
    setTimeout(() => {
        const form = document.getElementById('predictionForm');
        if (form) {
            form.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        
        // Automatically trigger prediction after values are loaded
        setTimeout(() => {
            console.log('Auto-triggering prediction with extracted values...');
            if (window.triggerPrediction) {
                window.triggerPrediction();
            } else {
                console.error('triggerPrediction not found on window!');
                showError('Error: Prediction function not available. Please refresh the page.');
            }
        }, 500);
    }, 300);
};

// Load history
async function loadHistory() {
    if (!currentUser) return;
    
    try {
        const response = await fetch('/api/predictions', {
            credentials: 'include'
        });
        const data = await response.json();
        
        if (response.ok && data.predictions) {
            displayHistory(data.predictions);
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

// Display history
function displayHistory(predictions) {
    const list = document.getElementById('historyList');
    
    if (predictions.length === 0) {
        list.innerHTML = '<p class="empty-state">No predictions yet. Make your first prediction!</p>';
        return;
    }
    
    list.innerHTML = predictions.map(pred => `
        <div class="history-item">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div>
                    <h3>Prediction #${pred.id}</h3>
                    <p style="color: #7f8c8d; margin: 10px 0;">
                        ${new Date(pred.prediction_date).toLocaleString()}
                    </p>
                    <div style="display: flex; gap: 15px; margin-top: 10px;">
                        <span>Anemia: <strong class="risk-${pred.anemia_risk.toLowerCase()}">${pred.anemia_risk}</strong></span>
                        <span>Neutropenia: <strong class="risk-${pred.neutropenia_risk.toLowerCase()}">${pred.neutropenia_risk}</strong></span>
                        <span>Thrombocytopenia: <strong class="risk-${pred.thrombocytopenia_risk.toLowerCase()}">${pred.thrombocytopenia_risk}</strong></span>
                    </div>
                    <p style="margin-top: 10px;">
                        <strong>Combined Severity:</strong> 
                        <span class="risk-${pred.combined_severity.toLowerCase()}">${pred.combined_severity}</span>
                    </p>
                </div>
                <div>
                    <a href="/api/export/pdf/${pred.id}" class="btn btn-outline" target="_blank">📄 PDF</a>
                </div>
            </div>
        </div>
    `).join('');
}

// Load reports
async function loadReports() {
    if (!currentUser) return;
    
    try {
        const response = await fetch('/api/reports', {
            credentials: 'include'
        });
        const data = await response.json();
        
        if (response.ok && data.reports) {
            // Display reports if needed
        }
    } catch (error) {
        console.error('Error loading reports:', error);
    }
}

// Load profile
async function loadProfile() {
    if (!currentUser) return;
    
    try {
        const response = await fetch('/api/auth/profile', {
            credentials: 'include'
        });
        const data = await response.json();
        
        if (response.ok && data.user) {
            document.getElementById('profileUsername').value = data.user.username;
            document.getElementById('profileEmail').value = data.user.email || '';
            document.getElementById('profileFullName').value = data.user.full_name || '';
        }
    } catch (error) {
        console.error('Error loading profile:', error);
    }
}

// Profile update
async function handleProfileUpdate(e) {
    e.preventDefault();
    
    const email = document.getElementById('profileEmail').value;
    const fullName = document.getElementById('profileFullName').value;
    
    try {
        const response = await fetch('/api/auth/profile', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrfToken() },
            credentials: 'include',
            body: JSON.stringify({ email, full_name: fullName })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            currentUser = data.user;
            document.getElementById('userName').textContent = currentUser.username;
            showError('Profile updated successfully!', 'success');
        } else {
            showError(data.error || 'Update failed');
        }
    } catch (error) {
        showError('Network error. Please try again.');
    }
}

// Export PDF
function exportPDF() {
    if (currentPredictionId) {
        window.open(`/api/export/pdf/${currentPredictionId}`, '_blank');
    }
}

// Share QR
function shareQR() {
    if (currentPredictionId) {
        window.open(`/api/share/qr/${currentPredictionId}`, '_blank');
    }
}

// Reset form
function resetForm() {
    document.getElementById('predictionForm').reset();
    document.getElementById('results').style.display = 'none';
    currentPredictionId = null;
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Show error/success/info message
function showError(message, type = 'error') {
    const errorDiv = document.getElementById('error');
    const errorText = document.getElementById('errorText');
    const messageTitle = document.getElementById('messageTitle');
    
    // Set message text
    errorText.textContent = message;
    
    // Set title and icon based on type
    let title = '';
    switch(type) {
        case 'success':
            title = '✅ Success';
            break;
        case 'info':
            title = 'ℹ️ Info';
            break;
        case 'error':
        default:
            title = '⚠️ Error';
            break;
    }
    
    if (messageTitle) {
        messageTitle.textContent = title;
    }
    
    // Set class and show
    errorDiv.className = `message-container ${type}`;
    errorDiv.style.display = 'block';
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 5000);
}

