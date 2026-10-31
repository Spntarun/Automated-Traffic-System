// State fetching interval
let globalState = {};
let congestionChart;

function initChart() {
    const ctx = document.getElementById('congestionChart').getContext('2d');
    congestionChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['North', 'South', 'East', 'West'],
            datasets: [{
                label: 'Vehicle Count',
                data: [0, 0, 0, 0],
                backgroundColor: [
                    'rgba(54, 162, 235, 0.6)',
                    'rgba(255, 99, 132, 0.6)',
                    'rgba(255, 206, 86, 0.6)',
                    'rgba(75, 192, 192, 0.6)'
                ],
                borderColor: [
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 99, 132, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(75, 192, 192, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });
}

// Initialize the chart immediately
initChart();

let fetchInterval = setInterval(fetchState, 1000);

async function fetchState() {
    try {
        const response = await fetch('/api/state');
        const data = await response.json();
        updateUI(data);
        
        if (congestionChart) {
            congestionChart.data.datasets[0].data = [
                data.lanes['N'].vehicle_count,
                data.lanes['S'].vehicle_count,
                data.lanes['E'].vehicle_count,
                data.lanes['W'].vehicle_count
            ];
            congestionChart.update();
        }
    } catch (error) {
        console.error('Error fetching state:', error);
    }
}

function updateUI(state) {
    // Update lane counters & Signals
    ['N', 'S', 'E', 'W'].forEach(lane => {
        const statEl = document.getElementById(`stat-${lane}`);
        if (statEl) {
            const vCount = state.lanes[lane].vehicle_count;
            const eCount = state.lanes[lane].emergency_count;
            const aCount = state.lanes[lane].accident_count || 0;
            
            statEl.querySelector('.v-count').innerText = vCount;
            statEl.querySelector('.e-count').innerText = eCount;
            const aCountEl = statEl.querySelector('.a-count');
            if (aCountEl) aCountEl.innerText = aCount;
            
            const isAccident = (aCount > 0) || (state.active_lane === lane && state.current_phase === 'AccidentResponse');
            const isEmergency = (eCount > 0) || (state.active_lane === lane && state.current_phase === 'Emergency');

            const alertEl = document.getElementById(`alert-${lane}`);

            if (isAccident) {
                statEl.classList.remove('emergency-active');
                statEl.classList.add('alert-active');
                if (alertEl) {
                    alertEl.className = 'alert-badge';
                    alertEl.innerText = '⚠️ ACCIDENT DETECTED!';
                    alertEl.style.display = 'block';
                }
            } else if (isEmergency) {
                statEl.classList.remove('alert-active');
                statEl.classList.add('emergency-active');
                if (alertEl) {
                    alertEl.className = 'alert-badge emergency-badge';
                    alertEl.innerText = '🚨 EMERGENCY VEHICLE DETECTED!';
                    alertEl.style.display = 'block';
                }
            } else {
                statEl.classList.remove('alert-active', 'emergency-active');
                if (alertEl) {
                    alertEl.style.display = 'none';
                }
            }
        }
        
        // Update Vehicle Traffic Lights
        const lightEl = document.getElementById(`light-${lane}`);
        const timerV = document.getElementById(`timer-v-${lane}`);
        if (lightEl) {
            lightEl.classList.remove('state-Red', 'state-Yellow', 'state-Green');
            lightEl.classList.add(`state-${state.lanes[lane].signal}`);
            
            if (state.lanes[lane].signal !== 'Red') {
                timerV.innerText = state.phase_timer;
                timerV.style.opacity = 1;
            } else {
                timerV.innerText = '--';
                timerV.style.opacity = 0.3;
            }
        }

        // Update Pedestrian Traffic Lights
        const pedLightEl = document.getElementById(`ped-${lane}`);
        const timerP = document.getElementById(`timer-p-${lane}`);
        if (pedLightEl) {
            pedLightEl.classList.remove('state-Red', 'state-Green');
            pedLightEl.classList.add(`state-${state.lanes[lane].pedestrian}`);
            
            if (state.lanes[lane].pedestrian !== 'Red') {
                timerP.innerText = state.phase_timer;
                timerP.style.opacity = 1;
            } else {
                timerP.innerText = '--';
                timerP.style.opacity = 0.3;
            }
        }
    });

    // Handle global popup
    const globalPopup = document.getElementById('global-alert-popup');
    const globalAlertText = document.getElementById('global-alert-text');
    
    if (state.current_phase === 'Emergency') {
        globalPopup.style.display = 'block';
        globalPopup.className = 'global-popup popup-emergency';
        globalAlertText.innerText = `🚑 EMERGENCY VEHICLE DETECTED IN ${state.active_lane} LANE - GREEN CORRIDOR ACTIVE`;
    } else if (state.current_phase === 'AccidentResponse') {
        globalPopup.style.display = 'block';
        globalPopup.className = 'global-popup popup-accident';
        globalAlertText.innerText = `⚠️ ACCIDENT DETECTED IN ${state.active_lane} LANE - TRAFFIC DIVERTED`;
    } else {
        globalPopup.style.display = 'none';
    }
    
    // Removed invalid current-mode element reference

    // Update Mode Indicator
    const modeIndicator = document.getElementById('mode-indicator');
    if (state.override_active) {
        modeIndicator.innerText = `Mode: Manual Override (${state.active_lane})`;
        modeIndicator.className = 'mode-indicator manual';
    } else if (state.pedestrian_override) {
        modeIndicator.innerText = `Mode: Pedestrian Override (${state.pedestrian_override})`;
        modeIndicator.className = 'mode-indicator manual';
    } else {
        modeIndicator.innerText = 'Mode: Auto Optimize';
        modeIndicator.className = 'mode-indicator';
    }
}

async function uploadMedia(lane) {
    const fileInput = document.getElementById(`upload-${lane}`);
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    fileInput.disabled = true;
    
    try {
        const response = await fetch(`/api/upload/${lane}`, {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        console.log(`Upload result for ${lane}:`, result);
        
        const imgEl = document.getElementById(`out-${lane}`);
        if (imgEl) {
            imgEl.src = `/static/outputs/${lane}.jpg?t=${new Date().getTime()}`;
            imgEl.style.display = 'block';
        }

        fetchState();
    } catch (error) {
        console.error('Upload error:', error);
    } finally {
        fileInput.disabled = false;
        fileInput.value = '';
    }
}

async function overrideSignal(lane) {
    try {
        await fetch('/api/override', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ lane: lane, active: true })
        });
        fetchState();
    } catch (error) { console.error(error); }
}

async function overridePedestrian(lane) {
    try {
        await fetch('/api/override_pedestrian', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ lane: lane })
        });
        fetchState();
    } catch (error) { console.error(error); }
}

async function resumeAuto() {
    try {
        await fetch('/api/override', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ lane: null, active: false })
        });
        fetchState();
    } catch (error) { console.error(error); }
}
