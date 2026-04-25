/**
 * LocalTranscript - Frontend JavaScript
 * Mit Zeit-Schätzung, Progress-Steps und Setup-Screen
 */

const API_BASE = '/api';
const electronBridge = window.electronApi || null;
const isElectron = !!electronBridge;

// DOM Elements - First Run (Electron only)
const firstRunSection = document.getElementById('firstRunSection');
const firstRunDefaultPath = document.getElementById('firstRunDefaultPath');
const firstRunDefaultBtn = document.getElementById('firstRunDefaultBtn');
const firstRunPickBtn = document.getElementById('firstRunPickBtn');

// DOM Elements - Setup
const setupSection = document.getElementById('setupSection');
const tokenInput = document.getElementById('tokenInput');
const toggleToken = document.getElementById('toggleToken');
const saveTokenBtn = document.getElementById('saveTokenBtn');

// DOM Elements - Upload
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const modelSelect = document.getElementById('modelSelect');
const languageSelect = document.getElementById('languageSelect');
const speakerRange = document.getElementById('speakerRange');
const clusterThreshold = document.getElementById('clusterThreshold');
const diarizeCheckbox = document.getElementById('diarizeCheckbox');
const speakerRangeGroup = document.getElementById('speakerRangeGroup');
const clusterThresholdGroup = document.getElementById('clusterThresholdGroup');
const startBtn = document.getElementById('startBtn');

const uploadSection = document.getElementById('uploadSection');
const progressSection = document.getElementById('progressSection');
const resultSection = document.getElementById('resultSection');
const errorSection = document.getElementById('errorSection');

const progressTitle = document.getElementById('progressTitle');
const progressPercent = document.getElementById('progressPercent');
const progressFill = document.getElementById('progressFill');
const progressMessage = document.getElementById('progressMessage');
const timeElapsed = document.getElementById('timeElapsed');
const timeEstimate = document.getElementById('timeEstimate');

const renameModal = document.getElementById('renameModal');
const renameList = document.getElementById('renameList');
const renameSaveBtn = document.getElementById('renameSaveBtn');
const renameCancelBtn = document.getElementById('renameCancelBtn');
const closeRenameBtn = document.getElementById('closeRenameBtn');

const glossaryModal = document.getElementById('glossaryModal');
const glossaryList = document.getElementById('glossaryList');
const glossarySearch = document.getElementById('glossarySearch');
const glossarySaveBtn = document.getElementById('glossarySaveBtn');
const glossaryCancelBtn = document.getElementById('glossaryCancelBtn');
const closeGlossaryBtn = document.getElementById('closeGlossaryBtn');

const resultList = document.getElementById('resultList');
const resultTitle = document.getElementById('resultTitle');
const resultSubtitle = document.getElementById('resultSubtitle');
const showRunBtn = document.getElementById('showRunBtn');
const batchIndicator = document.getElementById('batchIndicator');
const newBtn = document.getElementById('newBtn');
const retryBtn = document.getElementById('retryBtn');
const cancelBtn = document.getElementById('cancelBtn');
const errorMessage = document.getElementById('errorMessage');
const liveTranscript = document.getElementById('liveTranscript');
const transcriptText = document.getElementById('transcriptText');

// Step elements
const steps = {
    step1: document.getElementById('step1'),
    step2: document.getElementById('step2'),
    step3: document.getElementById('step3'),
    step4: document.getElementById('step4')
};

// State
let selectedFiles = [];
let currentJobId = null;
let pollInterval = null;
let timerInterval = null;
let startTime = null;
let estimatedDuration = null;
let batchResults = [];       // {filename, jobId, status, error}
let batchIndex = 0;          // index of currently processing file
let batchAborted = false;
let currentRunFolder = null; // Electron run folder path for the current batch
let transcriptsRoot = null;  // Cached config from Electron
let renameContext = null;    // {jobId, audio, currentRow} while modal is open
let glossaryContext = null;  // {jobId, candidates, result} while modal is open

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    setupEventListeners();
    if (isElectron) {
        electronBridge.onConfigChanged((cfg) => {
            transcriptsRoot = cfg && cfg.transcriptsRoot ? cfg.transcriptsRoot : null;
        });
        const cfg = await electronBridge.getConfig().catch(() => ({}));
        transcriptsRoot = cfg && cfg.transcriptsRoot ? cfg.transcriptsRoot : null;
    }
    await routeStartScreen();
});

async function routeStartScreen() {
    // First-run setup is Electron-only.
    if (isElectron && !transcriptsRoot) {
        showFirstRun();
        return;
    }
    await checkSetup();
}

function hideAllSections() {
    if (firstRunSection) firstRunSection.hidden = true;
    setupSection.hidden = true;
    uploadSection.hidden = true;
    progressSection.hidden = true;
    resultSection.hidden = true;
    errorSection.hidden = true;
}

function showFirstRun() {
    hideAllSections();
    firstRunSection.hidden = false;
    if (firstRunDefaultPath) {
        firstRunDefaultPath.textContent = '~/Documents/LocalTranscript';
    }
}

async function checkSetup() {
    // SpeechBrain-based diarization needs no HuggingFace token, so the legacy
    // setup-section is bypassed. Kept in HTML for graceful degradation only.
    hideAllSections();
    uploadSection.hidden = false;
    try {
        await loadModels();
    } catch (error) {
        console.error('Model load failed:', error);
    }
}

async function loadModels() {
    try {
        const response = await fetch(`${API_BASE}/models`);
        const data = await response.json();

        modelSelect.innerHTML = '';
        data.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.name;

            // Add recommendation for models
            let label = capitalize(model.name);
            if (model.name === 'large-v3-turbo') {
                label += ' (empfohlen)';
            } else if (model.name === 'large-v3') {
                label += ' (beste Qualität)';
            } else if (model.name === 'medium') {
                label += ' (schneller)';
            }
            label += ` - ${formatSize(model.size_mb)}`;

            option.textContent = label;
            // Default: large-v3-turbo
            if (model.name === 'large-v3-turbo') option.selected = true;
            modelSelect.appendChild(option);
        });

        // Fallback: if no turbo, select large-v3, then medium
        if (!modelSelect.value) {
            const largeOption = modelSelect.querySelector('[value="large-v3"]');
            const mediumOption = modelSelect.querySelector('[value="medium"]');
            if (largeOption) largeOption.selected = true;
            else if (mediumOption) mediumOption.selected = true;
        }
    } catch (error) {
        console.error('Failed to load models:', error);
    }
}

function setupEventListeners() {
    // First-run (Electron only)
    if (firstRunDefaultBtn) {
        firstRunDefaultBtn.addEventListener('click', () => pickTranscriptsRoot('default'));
    }
    if (firstRunPickBtn) {
        firstRunPickBtn.addEventListener('click', () => pickTranscriptsRoot('pick'));
    }
    if (closeRenameBtn) closeRenameBtn.addEventListener('click', closeRenameModal);
    if (renameCancelBtn) renameCancelBtn.addEventListener('click', closeRenameModal);
    if (renameSaveBtn) renameSaveBtn.addEventListener('click', saveSpeakerNames);
    if (renameModal) {
        renameModal.addEventListener('click', (e) => {
            if (e.target === renameModal) closeRenameModal();
        });
    }

    if (closeGlossaryBtn) closeGlossaryBtn.addEventListener('click', closeGlossaryModal);
    if (glossaryCancelBtn) glossaryCancelBtn.addEventListener('click', closeGlossaryModal);
    if (glossarySaveBtn) glossarySaveBtn.addEventListener('click', saveGlossaryRenames);
    if (glossarySearch) glossarySearch.addEventListener('input', filterGlossaryList);
    if (glossaryModal) {
        glossaryModal.addEventListener('click', (e) => {
            if (e.target === glossaryModal) closeGlossaryModal();
        });
    }

    if (showRunBtn) {
        showRunBtn.addEventListener('click', () => {
            if (currentRunFolder && electronBridge) {
                electronBridge.openFolder(currentRunFolder).catch(err => {
                    alert('Konnte Ordner nicht öffnen: ' + err.message);
                });
            }
        });
    }

    // Setup screen
    if (saveTokenBtn) {
        saveTokenBtn.addEventListener('click', saveToken);
    }
    if (toggleToken) {
        toggleToken.addEventListener('click', toggleTokenVisibility);
    }
    if (tokenInput) {
        tokenInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') saveToken();
        });
    }

    // Drag and drop
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);

    fileInput.addEventListener('change', handleFileSelect);
    startBtn.addEventListener('click', startTranscription);
    if (diarizeCheckbox) {
        diarizeCheckbox.addEventListener('change', updateDiarizeState);
        updateDiarizeState();
    }
    newBtn.addEventListener('click', resetUI);
    retryBtn.addEventListener('click', resetUI);
    cancelBtn.addEventListener('click', cancelJob);
}

async function cancelJob() {
    if (!currentJobId) return;

    const msg = selectedFiles.length > 1
        ? 'Batch wirklich abbrechen? Bereits fertige Dateien bleiben erhalten.'
        : 'Transkription wirklich abbrechen?';
    if (!confirm(msg)) return;

    batchAborted = true;
    stopPolling();

    try {
        await fetch(`${API_BASE}/jobs/${currentJobId}`, {
            method: 'DELETE'
        });
    } catch (error) {
        console.error('Cancel error:', error);
    }

    finishBatch();
}

function toggleTokenVisibility() {
    if (tokenInput.type === 'password') {
        tokenInput.type = 'text';
        toggleToken.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/>
            <line x1="1" y1="1" x2="23" y2="23"/>
        </svg>`;
    } else {
        tokenInput.type = 'password';
        toggleToken.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
            <circle cx="12" cy="12" r="3"/>
        </svg>`;
    }
}

async function saveToken() {
    const token = tokenInput.value.trim();

    if (!token) {
        alert('Bitte Token eingeben');
        return;
    }

    if (!token.startsWith('hf_')) {
        alert('Token muss mit "hf_" beginnen');
        return;
    }

    saveTokenBtn.disabled = true;
    saveTokenBtn.textContent = 'Speichere...';

    try {
        const response = await fetch(`${API_BASE}/setup/token`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ token: token })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Token konnte nicht gespeichert werden');
        }

        // Success - show upload section
        setupSection.hidden = true;
        uploadSection.hidden = false;
        loadModels();

    } catch (error) {
        alert('Fehler: ' + error.message);
        saveTokenBtn.disabled = false;
        saveTokenBtn.textContent = 'Token speichern & starten';
    }
}

async function pickTranscriptsRoot(mode) {
    if (!electronBridge) return;
    try {
        const result = await electronBridge.setTranscriptsRoot(mode);
        if (result && result.canceled) return;
        if (result && result.transcriptsRoot) {
            transcriptsRoot = result.transcriptsRoot;
            await checkSetup();
        }
    } catch (err) {
        alert('Speicherort konnte nicht gesetzt werden: ' + (err.message || err));
    }
}

function pad2(n) { return n < 10 ? '0' + n : '' + n; }

function makeRunFolderName() {
    const d = new Date();
    const yy = pad2(d.getFullYear() % 100);
    const mm = pad2(d.getMonth() + 1);
    const dd = pad2(d.getDate());
    const hh = pad2(d.getHours());
    const mi = pad2(d.getMinutes());
    return `${yy}${mm}${dd}_${hh}${mi}`;
}

function audioBasename(filename) {
    return filename.replace(/\.[^.]+$/, '');
}

function updateDiarizeState() {
    const enabled = diarizeCheckbox.checked;
    [speakerRangeGroup, clusterThresholdGroup].forEach(el => {
        if (!el) return;
        el.classList.toggle('disabled', !enabled);
        const control = el.querySelector('select, input');
        if (control) control.disabled = !enabled;
    });
}

function handleDragOver(e) {
    e.preventDefault();
    dropZone.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    addFiles(e.dataTransfer.files);
}

function handleFileSelect(e) {
    addFiles(e.target.files);
}

const VALID_EXTS = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.webm'];

function addFiles(fileList) {
    const rejected = [];
    for (const file of fileList) {
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!VALID_EXTS.includes(ext)) {
            rejected.push(file.name);
            continue;
        }
        // Skip duplicates (same name + size)
        if (selectedFiles.some(f => f.name === file.name && f.size === file.size)) {
            continue;
        }
        selectedFiles.push(file);
    }

    if (rejected.length) {
        alert(`Ungültiges Dateiformat:\n${rejected.join('\n')}\n\nErlaubt: ${VALID_EXTS.join(', ')}`);
    }

    fileInput.value = '';
    renderFileList();
}

function renderFileList() {
    fileList.innerHTML = '';
    if (selectedFiles.length === 0) {
        fileList.hidden = true;
        dropZone.hidden = false;
        startBtn.disabled = true;
        startBtn.textContent = 'Transkription starten';
        estimatedDuration = null;
        return;
    }

    fileList.hidden = false;
    dropZone.hidden = false;  // keep drop zone visible so user can add more

    selectedFiles.forEach((file, idx) => {
        const li = document.createElement('li');
        li.innerHTML = `
            <span class="file-name"></span>
            <span class="file-size"></span>
            <button class="btn-clear" type="button" aria-label="Entfernen">&times;</button>
        `;
        li.querySelector('.file-name').textContent = file.name;
        li.querySelector('.file-size').textContent = formatSize(file.size / (1024 * 1024));
        li.querySelector('.btn-clear').addEventListener('click', () => {
            selectedFiles.splice(idx, 1);
            renderFileList();
        });
        fileList.appendChild(li);
    });

    startBtn.disabled = false;
    startBtn.textContent = selectedFiles.length === 1
        ? 'Transkription starten'
        : `${selectedFiles.length} Dateien transkribieren`;

    const totalBytes = selectedFiles.reduce((sum, f) => sum + f.size, 0);
    estimatedDuration = estimateProcessingTime(totalBytes);
}

function estimateProcessingTime(fileSizeBytes) {
    // Rough estimates based on M3 Pro performance:
    // - 1 MB audio ≈ 1 minute of audio
    // - Whisper processes ~0.5x realtime with large model
    // - Diarization adds ~0.3x
    // So total: ~1.6 minutes per MB for large model

    const fileSizeMB = fileSizeBytes / (1024 * 1024);
    const model = modelSelect.value;

    // Processing speed factor (smaller = faster)
    let speedFactor = model.includes('large') ? 1.6 : 0.8;

    // Estimated minutes
    const estimatedMinutes = fileSizeMB * speedFactor;

    // Return in seconds, minimum 30 seconds
    return Math.max(30, estimatedMinutes * 60);
}

async function startTranscription() {
    if (selectedFiles.length === 0) return;

    // Show progress
    uploadSection.hidden = true;
    progressSection.hidden = false;
    resultSection.hidden = true;
    errorSection.hidden = true;

    batchResults = [];
    batchIndex = 0;
    batchAborted = false;
    currentRunFolder = null;

    // Create the run folder up-front so partial results are saved if a later file fails.
    if (isElectron && transcriptsRoot) {
        try {
            const res = await electronBridge.createRunFolder(makeRunFolderName());
            currentRunFolder = res && res.path ? res.path : null;
        } catch (err) {
            alert('Konnte Run-Ordner nicht anlegen: ' + (err.message || err) + '\n\nDie Transkription läuft, aber Autosave ist deaktiviert.');
            currentRunFolder = null;
        }
    }

    startTime = Date.now();
    startTimer();

    await processNextFile();
}

async function processNextFile() {
    if (batchAborted) {
        finishBatch();
        return;
    }

    if (batchIndex >= selectedFiles.length) {
        finishBatch();
        return;
    }

    const file = selectedFiles[batchIndex];
    updateBatchIndicator();

    // Reset steps + progress for this file
    resetSteps();
    setStepActive(1);
    updateProgress(0, `Lade ${file.name} hoch...`);
    liveTranscript.hidden = true;
    transcriptText.textContent = '';

    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('model', modelSelect.value);
        formData.append('language', languageSelect.value);
        formData.append('speaker_range', speakerRange.value);
        formData.append('cluster_threshold', clusterThreshold.value);
        formData.append('diarize', diarizeCheckbox && diarizeCheckbox.checked ? 'true' : 'false');

        const response = await fetch(`${API_BASE}/transcribe`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || 'Upload fehlgeschlagen');
        }

        const data = await response.json();
        currentJobId = data.job_id;
        setStepCompleted(1);
        startPolling();

    } catch (error) {
        batchResults.push({
            filename: file.name,
            jobId: null,
            status: 'failed',
            error: error.message
        });
        batchIndex++;
        processNextFile();
    }
}

function updateBatchIndicator() {
    if (!batchIndicator) return;
    if (selectedFiles.length <= 1) {
        batchIndicator.hidden = true;
        return;
    }
    batchIndicator.hidden = false;
    const current = selectedFiles[batchIndex];
    batchIndicator.textContent = `Datei ${batchIndex + 1} von ${selectedFiles.length}: ${current ? current.name : ''}`;
}

function openRenameModal(result) {
    if (!renameModal || !result.speakers || result.speakers.length === 0) return;

    renameContext = { jobId: result.jobId, audio: null, currentRow: null, result };

    renameList.innerHTML = '';
    result.speakers.forEach((speakerLabel, idx) => {
        const li = document.createElement('li');
        li.dataset.speaker = speakerLabel;

        const playBtn = document.createElement('button');
        playBtn.type = 'button';
        playBtn.className = 'rename-play';
        playBtn.textContent = '▶ Anhören';
        playBtn.addEventListener('click', () => toggleSpeakerPlayback(speakerLabel, playBtn));
        li.appendChild(playBtn);

        const nameInput = document.createElement('input');
        nameInput.type = 'text';
        nameInput.className = 'rename-name';
        nameInput.placeholder = `Speaker ${idx + 1}`;
        // Pre-fill with the current display name (or default if equal to default)
        const current = (result.speakerNames && result.speakerNames[speakerLabel]) || `Speaker ${idx + 1}`;
        if (current !== `Speaker ${idx + 1}`) {
            nameInput.value = current;
        }
        nameInput.dataset.speaker = speakerLabel;
        li.appendChild(nameInput);

        renameList.appendChild(li);
    });

    renameModal.hidden = false;
}

function closeRenameModal() {
    if (renameContext && renameContext.audio) {
        renameContext.audio.pause();
        renameContext.audio = null;
    }
    if (renameModal) renameModal.hidden = true;
    renameContext = null;
}

function toggleSpeakerPlayback(speakerLabel, btn) {
    if (!renameContext) return;

    // Stop any currently playing audio
    if (renameContext.audio) {
        renameContext.audio.pause();
        renameContext.audio = null;
    }
    document.querySelectorAll('.rename-play.playing').forEach(b => {
        b.classList.remove('playing');
        b.textContent = '▶ Anhören';
    });

    if (renameContext.currentRow === speakerLabel) {
        // Was playing this one — toggle off
        renameContext.currentRow = null;
        return;
    }

    const audio = new Audio(`${API_BASE}/jobs/${renameContext.jobId}/speaker/${encodeURIComponent(speakerLabel)}/sample`);
    audio.loop = true;
    audio.play().catch(err => {
        alert('Konnte Audio-Probe nicht abspielen: ' + (err.message || err));
    });
    renameContext.audio = audio;
    renameContext.currentRow = speakerLabel;
    btn.classList.add('playing');
    btn.textContent = '■ Stop';
}

async function saveSpeakerNames() {
    if (!renameContext) return;

    // Collect names
    const names = {};
    renameList.querySelectorAll('input.rename-name').forEach(input => {
        const sp = input.dataset.speaker;
        const value = input.value.trim();
        if (sp && value) names[sp] = value;
    });

    renameSaveBtn.disabled = true;
    renameSaveBtn.textContent = 'Speichere...';

    try {
        const response = await fetch(`${API_BASE}/jobs/${renameContext.jobId}/rename-speakers`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ names })
        });
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || 'Speichern fehlgeschlagen');
        }
        const data = await response.json();

        // If we autosaved into the run folder, refresh those copies too.
        if (currentRunFolder && isElectron && Object.keys(data.renamed || {}).length > 0) {
            try {
                const job = await fetch(`${API_BASE}/jobs/${renameContext.jobId}`).then(r => r.json());
                await autosaveJob(job, currentRunFolder);
            } catch (err) {
                console.warn('Re-copy after rename failed:', err);
            }
        }

        // Update local state for this result
        if (renameContext.result) {
            renameContext.result.speakerNames = data.speaker_names || {};
        }

        closeRenameModal();
    } catch (error) {
        alert('Fehler: ' + (error.message || error));
        renameSaveBtn.disabled = false;
        renameSaveBtn.textContent = 'Speichern';
    } finally {
        renameSaveBtn.disabled = false;
        renameSaveBtn.textContent = 'Speichern';
    }
}

async function openGlossaryModal(result) {
    if (!glossaryModal) return;
    glossaryContext = { jobId: result.jobId, candidates: [], result };

    glossaryList.innerHTML = '';
    glossaryList.appendChild(makeLoadingRow('Lade Begriffe…'));
    if (glossarySearch) glossarySearch.value = '';
    glossaryModal.hidden = false;

    try {
        const response = await fetch(`${API_BASE}/jobs/${result.jobId}/glossary`);
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || 'Glossar konnte nicht geladen werden');
        }
        const data = await response.json();
        glossaryContext.candidates = data.candidates || [];
        renderGlossaryList();
    } catch (err) {
        glossaryList.innerHTML = '';
        glossaryList.appendChild(makeLoadingRow('Fehler: ' + (err.message || err)));
    }
}

function makeLoadingRow(msg) {
    const li = document.createElement('li');
    li.className = 'glossary-row';
    li.style.gridTemplateColumns = '1fr';
    li.style.color = 'var(--text-secondary)';
    li.textContent = msg;
    return li;
}

function renderGlossaryList() {
    if (!glossaryContext) return;
    glossaryList.innerHTML = '';

    if (glossaryContext.candidates.length === 0) {
        glossaryList.appendChild(makeLoadingRow('Keine Begriffe gefunden.'));
        return;
    }

    glossaryContext.candidates.forEach((c) => {
        const li = document.createElement('li');
        li.className = 'glossary-row';
        li.dataset.term = c.term;

        const count = document.createElement('span');
        count.className = 'glossary-count';
        count.textContent = '×' + c.count;
        li.appendChild(count);

        const orig = document.createElement('span');
        orig.className = 'glossary-original';
        const kindBadge = document.createElement('span');
        kindBadge.className = 'glossary-kind kind-' + c.kind;
        kindBadge.textContent = c.kind;
        orig.appendChild(kindBadge);
        const termText = document.createElement('span');
        termText.textContent = c.term;
        orig.appendChild(termText);
        li.appendChild(orig);

        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'rename-name';
        input.placeholder = c.term;
        input.dataset.original = c.term;
        li.appendChild(input);

        glossaryList.appendChild(li);
    });
}

function filterGlossaryList() {
    if (!glossaryContext || !glossarySearch) return;
    const q = glossarySearch.value.trim().toLowerCase();
    glossaryList.querySelectorAll('li').forEach((li) => {
        const term = (li.dataset.term || '').toLowerCase();
        li.hidden = q.length > 0 && !term.includes(q);
    });
}

function closeGlossaryModal() {
    if (glossaryModal) glossaryModal.hidden = true;
    glossaryContext = null;
}

async function saveGlossaryRenames() {
    if (!glossaryContext) return;

    const renames = {};
    glossaryList.querySelectorAll('input.rename-name').forEach((input) => {
        const original = input.dataset.original;
        const value = input.value.trim();
        if (original && value && value !== original) {
            renames[original] = value;
        }
    });

    if (Object.keys(renames).length === 0) {
        closeGlossaryModal();
        return;
    }

    glossarySaveBtn.disabled = true;
    glossarySaveBtn.textContent = 'Speichere...';

    try {
        const response = await fetch(`${API_BASE}/jobs/${glossaryContext.jobId}/rename-terms`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ renames })
        });
        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || 'Speichern fehlgeschlagen');
        }

        // Re-copy updated files into the run folder if Electron mode
        if (currentRunFolder && isElectron) {
            try {
                const job = await fetch(`${API_BASE}/jobs/${glossaryContext.jobId}`).then(r => r.json());
                await autosaveJob(job, currentRunFolder);
            } catch (err) {
                console.warn('Re-copy after term-rename failed:', err);
            }
        }

        closeGlossaryModal();
    } catch (error) {
        alert('Fehler: ' + (error.message || error));
    } finally {
        glossarySaveBtn.disabled = false;
        glossarySaveBtn.textContent = 'Speichern';
    }
}

async function autosaveJob(job, runFolder) {
    const baseName = audioBasename(job.filename);
    const audioExt = (job.filename.match(/\.[^.]+$/) || ['.bin'])[0];
    const files = [];

    if (job.upload_path) {
        files.push({ src: job.upload_path, name: `${baseName}${audioExt}` });
    }
    if (job.output_path) {
        files.push({ src: job.output_path, name: `${baseName}.vtt` });
    }
    if (job.txt_path) {
        files.push({ src: job.txt_path, name: `${baseName}.txt` });
    }
    if (job.csv_path) {
        files.push({ src: job.csv_path, name: `${baseName}.csv` });
    }

    if (files.length === 0) return { copied: [] };
    return await electronBridge.copyToRun(runFolder, files);
}

function finishBatch() {
    stopPolling();
    stopTimer();
    currentJobId = null;
    progressSection.hidden = true;

    if (batchAborted && batchResults.length === 0) {
        resetUI();
        return;
    }

    renderResults();
    resultSection.hidden = false;
}

function renderResults() {
    const successCount = batchResults.filter(r => r.status === 'completed').length;
    const failCount = batchResults.filter(r => r.status === 'failed').length;

    if (batchResults.length === 1) {
        resultTitle.textContent = batchResults[0].status === 'completed'
            ? 'Transkription fertig!'
            : 'Transkription fehlgeschlagen';
    } else {
        let title = `${successCount} von ${batchResults.length} fertig`;
        if (failCount > 0) title += ` (${failCount} fehlgeschlagen)`;
        resultTitle.textContent = title;
    }

    // Subtitle (Electron mode): show run-folder path and any errors
    if (resultSubtitle) {
        if (isElectron && currentRunFolder) {
            const autosaveErrors = batchResults.filter(r => r.autosaveError);
            const lines = [`Gespeichert in: ${currentRunFolder}`];
            if (failCount > 0) lines.push(`${failCount} Datei(en) fehlgeschlagen.`);
            if (autosaveErrors.length > 0) {
                lines.push(`${autosaveErrors.length} Datei(en) konnten nicht autosaved werden.`);
            }
            resultSubtitle.textContent = lines.join('  ·  ');
            resultSubtitle.hidden = false;
        } else {
            resultSubtitle.hidden = true;
            resultSubtitle.textContent = '';
        }
    }
    if (showRunBtn) showRunBtn.hidden = !(isElectron && currentRunFolder);

    resultList.hidden = false;
    resultList.innerHTML = '';
    batchResults.forEach(res => {
        const li = document.createElement('li');
        if (res.status !== 'completed') li.classList.add('result-failed');

        const name = document.createElement('span');
        name.className = 'result-name';
        name.textContent = res.filename;
        li.appendChild(name);

        if (res.status === 'completed') {
            const actions = document.createElement('span');
            actions.className = 'result-downloads';
            // Show "Korrigieren" only when speakers exist (skip diarize=off jobs)
            if (res.speakerCount && res.speakerCount >= 1) {
                const correct = document.createElement('button');
                correct.type = 'button';
                correct.className = 'btn-correct';
                correct.textContent = 'Sprecher';
                correct.addEventListener('click', () => openRenameModal(res));
                actions.appendChild(correct);
            }
            const terms = document.createElement('button');
            terms.type = 'button';
            terms.className = 'btn-correct';
            terms.textContent = 'Begriffe';
            terms.addEventListener('click', () => openGlossaryModal(res));
            actions.appendChild(terms);
            li.appendChild(actions);
            // In Electron mode the files are already in the run folder, so
            // skip per-file VTT/CSV/TXT download buttons. Browser-mode keeps them.
            if (!(isElectron && currentRunFolder)) {
                const downloads = document.createElement('span');
                downloads.className = 'result-downloads';

                const txt = document.createElement('a');
                txt.className = 'btn-txt';
                txt.textContent = 'TXT';
                txt.addEventListener('click', () => saveFileForJob(res.jobId, 'txt'));
                downloads.appendChild(txt);

                const vtt = document.createElement('a');
                vtt.className = 'btn-vtt';
                vtt.textContent = 'VTT';
                vtt.addEventListener('click', () => saveFileForJob(res.jobId, 'vtt'));
                downloads.appendChild(vtt);

                const csv = document.createElement('a');
                csv.textContent = 'CSV';
                csv.addEventListener('click', () => saveFileForJob(res.jobId, 'csv'));
                downloads.appendChild(csv);

                li.appendChild(downloads);
            }
        } else {
            const status = document.createElement('span');
            status.className = 'result-status';
            status.textContent = res.error || 'Fehlgeschlagen';
            li.appendChild(status);
        }

        resultList.appendChild(li);
    });
}

function startTimer() {
    updateTimer();
    timerInterval = setInterval(updateTimer, 1000);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

function updateTimer() {
    if (!startTime) return;

    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    timeElapsed.textContent = formatTime(elapsed);

    if (estimatedDuration) {
        timeEstimate.textContent = formatTime(Math.round(estimatedDuration));
    }
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function formatSize(mb) {
    if (mb >= 1024) {
        return `${(mb / 1024).toFixed(1)} GB`;
    }
    return `${mb.toFixed(1)} MB`;
}

function startPolling() {
    pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/jobs/${currentJobId}`);
            const job = await response.json();

            updateProgress(job.progress, job.message);
            updateStepsFromStatus(job.status, job.progress);

            // Show live transcript
            if (job.partial_text) {
                liveTranscript.hidden = false;
                transcriptText.textContent = job.partial_text;
                // Auto-scroll to bottom
                liveTranscript.scrollTop = liveTranscript.scrollHeight;
            }

            if (job.status === 'completed') {
                stopPolling();
                setAllStepsCompleted();
                const result = {
                    filename: selectedFiles[batchIndex].name,
                    jobId: currentJobId,
                    status: 'completed',
                    speakers: job.speakers || [],
                    speakerCount: (job.speakers || []).length,
                    speakerNames: job.speaker_names || {}
                };
                if (currentRunFolder) {
                    try {
                        await autosaveJob(job, currentRunFolder);
                        result.autosaved = true;
                    } catch (err) {
                        console.error('Autosave failed:', err);
                        result.autosaveError = err.message || String(err);
                    }
                }
                batchResults.push(result);
                currentJobId = null;
                batchIndex++;
                processNextFile();
            } else if (job.status === 'failed') {
                stopPolling();
                batchResults.push({
                    filename: selectedFiles[batchIndex].name,
                    jobId: currentJobId,
                    status: 'failed',
                    error: job.error || 'Verarbeitung fehlgeschlagen'
                });
                currentJobId = null;
                batchIndex++;
                processNextFile();
            }

        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 1000);
}

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

function updateProgress(percent, message) {
    progressPercent.textContent = `${percent}%`;
    progressFill.style.width = `${percent}%`;
    progressMessage.textContent = message;

    // Update title based on message from backend (shows speaker count etc.)
    if (message && message.length > 0) {
        // Use backend message if it contains useful info
        if (message.includes('Sprecher') || message.includes('Transkri') || message.includes('Merge')) {
            progressTitle.textContent = message;
        } else if (percent < 10) {
            progressTitle.textContent = 'Lade hoch...';
        } else if (percent < 30) {
            progressTitle.textContent = 'Erkenne Sprecher...';
        } else if (percent < 80) {
            progressTitle.textContent = 'Transkribiere...';
        } else {
            progressTitle.textContent = 'Erstelle VTT...';
        }
    }

    // Update time estimate based on progress
    if (percent > 0 && percent < 100 && startTime) {
        const elapsed = (Date.now() - startTime) / 1000;
        const estimatedTotal = (elapsed / percent) * 100;
        estimatedDuration = estimatedTotal;
    }
}

function resetSteps() {
    Object.values(steps).forEach(step => {
        step.classList.remove('active', 'completed');
    });
}

function setStepActive(stepNum) {
    const step = steps[`step${stepNum}`];
    if (step) {
        step.classList.add('active');
        step.classList.remove('completed');
    }
}

function setStepCompleted(stepNum) {
    const step = steps[`step${stepNum}`];
    if (step) {
        step.classList.remove('active');
        step.classList.add('completed');
    }
}

function setAllStepsCompleted() {
    Object.values(steps).forEach(step => {
        step.classList.remove('active');
        step.classList.add('completed');
    });
}

function updateStepsFromStatus(status, progress) {
    // Step 1: Upload (always completed if we're polling)
    setStepCompleted(1);

    // Step 2: Speaker Diarization (10-30%)
    if (progress >= 10 && progress < 30) {
        setStepActive(2);
    } else if (progress >= 30) {
        setStepCompleted(2);
    }

    // Step 3: Transcription (30-80%)
    if (progress >= 30 && progress < 80) {
        setStepActive(3);
    } else if (progress >= 80) {
        setStepCompleted(3);
    }

    // Step 4: Merge (80-100%)
    if (progress >= 80 && progress < 100) {
        setStepActive(4);
    } else if (progress >= 100) {
        setStepCompleted(4);
    }
}

// Save file with native dialog (PyWebView) or fallback to Downloads
async function saveFileForJob(jobId, type) {
    try {
        const response = await fetch(`${API_BASE}/jobs/${jobId}`);
        const job = await response.json();

        const sourcePathByType = {
            vtt: job.output_path,
            csv: job.csv_path,
            txt: job.txt_path
        };
        const suffixByType = {
            vtt: '_transkript.vtt',
            csv: '_sprecher.csv',
            txt: '_transkript.txt'
        };
        const endpointByType = {
            vtt: 'save-vtt',
            csv: 'save-csv',
            txt: 'save-txt'
        };

        const sourcePath = sourcePathByType[type];
        const originalName = job.filename.replace(/\.[^.]+$/, '');
        const defaultName = `${originalName}${suffixByType[type]}`;

        if (window.pywebview && window.pywebview.api) {
            const result = await window.pywebview.api.save_file(sourcePath, defaultName, type);
            if (result.success) {
                alert(`Gespeichert:\n${result.path}`);
            } else if (result.error !== 'Abgebrochen') {
                alert('Fehler: ' + result.error);
            }
        } else {
            const saveResponse = await fetch(`${API_BASE}/jobs/${jobId}/${endpointByType[type]}`, {
                method: 'POST'
            });

            if (!saveResponse.ok) {
                const error = await saveResponse.json();
                throw new Error(error.detail || 'Speichern fehlgeschlagen');
            }

            const saveResult = await saveResponse.json();
            alert(`Gespeichert in Downloads:\n${saveResult.filename}`);
        }
    } catch (error) {
        console.error('Save error:', error);
        alert('Speichern fehlgeschlagen: ' + error.message);
    }
}

function resetUI() {
    stopPolling();
    stopTimer();
    currentJobId = null;
    selectedFiles = [];
    fileInput.value = '';
    startTime = null;
    estimatedDuration = null;
    batchResults = [];
    batchIndex = 0;
    batchAborted = false;
    currentRunFolder = null;

    if (resultSubtitle) {
        resultSubtitle.hidden = true;
        resultSubtitle.textContent = '';
    }
    if (showRunBtn) showRunBtn.hidden = true;

    uploadSection.hidden = false;
    progressSection.hidden = true;
    resultSection.hidden = true;
    errorSection.hidden = true;

    renderFileList();
    if (batchIndicator) {
        batchIndicator.hidden = true;
        batchIndicator.textContent = '';
    }

    liveTranscript.hidden = true;
    transcriptText.textContent = '';

    resetSteps();
    updateProgress(0, '');
    timeElapsed.textContent = '00:00';
    timeEstimate.textContent = '--:--';
}

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

// Info Modal
const infoBtn = document.getElementById('infoBtn');
const infoModal = document.getElementById('infoModal');
const closeInfoBtn = document.getElementById('closeInfoBtn');
const infoContent = document.getElementById('infoContent');

if (infoBtn) {
    infoBtn.addEventListener('click', showInfoModal);
}

if (closeInfoBtn) {
    closeInfoBtn.addEventListener('click', hideInfoModal);
}

if (infoModal) {
    infoModal.addEventListener('click', (e) => {
        if (e.target === infoModal) {
            hideInfoModal();
        }
    });
}

// Close modal with Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && infoModal && !infoModal.hidden) {
        hideInfoModal();
    }
});

async function showInfoModal() {
    infoModal.hidden = false;
    infoContent.innerHTML = '<div class="loading">Lade Informationen...</div>';

    try {
        const [infoRes, cfg] = await Promise.all([
            fetch(`${API_BASE}/info`).then(r => r.json()),
            isElectron ? electronBridge.getConfig().catch(() => ({})) : Promise.resolve({}),
        ]);
        renderInfoContent(infoRes, cfg);
    } catch (error) {
        infoContent.innerHTML = `<div class="error-content"><p>Fehler beim Laden: ${error.message}</p></div>`;
    }
}

function hideInfoModal() {
    infoModal.hidden = true;
}

function renderInfoContent(info, config) {
    config = config || {};
    const transcriptsRootHtml = (isElectron && config.transcriptsRoot)
        ? `
        <div class="info-section">
            <h3>Speicherort</h3>
            <div class="info-grid">
                <div class="info-row">
                    <span class="info-label">Transkript-Ordner</span>
                    <span class="info-value">
                        <a href="#" id="infoOpenFolder"><code>${config.transcriptsRoot}</code></a>
                    </span>
                </div>
                <div class="info-row">
                    <span class="info-label">Run-Format</span>
                    <span class="info-value"><code>yymmdd_hhmm/</code> — Audio + VTT + TXT + CSV</span>
                </div>
            </div>
        </div>
        `
        : '';

    const html = `
        ${transcriptsRootHtml}
        <div class="info-section">
            <h3>App & Entwickler</h3>
            <div class="info-grid">
                <div class="info-row">
                    <span class="info-label">Version</span>
                    <span class="info-value">${info.app.version}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Entwickler</span>
                    <span class="info-value">${info.app.developer}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Kontakt</span>
                    <span class="info-value"><a href="mailto:${info.app.contact}">${info.app.contact}</a></span>
                </div>
                <div class="info-row">
                    <span class="info-label">Lizenz</span>
                    <span class="info-value"><span class="info-badge">${info.app.license}</span></span>
                </div>
                <div class="info-row">
                    <span class="info-label">GitHub</span>
                    <span class="info-value">
                        <a href="#" id="infoOpenRepo">${info.app.repository}</a>
                    </span>
                </div>
            </div>
        </div>

        <div class="info-section">
            <h3>API Zugriff</h3>
            <div class="info-grid">
                <div class="info-row">
                    <span class="info-label">Lokal</span>
                    <span class="info-value"><code>${info.api.local}</code></span>
                </div>
            </div>
            <div style="margin-top: 12px;">
                ${Object.entries(info.api.endpoints).map(([endpoint, desc]) => `
                    <div class="api-endpoint">
                        <code>${endpoint}</code>
                        <span>— ${desc}</span>
                    </div>
                `).join('')}
            </div>
        </div>

        <div class="info-section">
            <h3>Technologie</h3>
            <div class="info-grid">
                <div class="info-row">
                    <span class="info-label">Transkription</span>
                    <span class="info-value">${info.technology.transcription.name} (${info.technology.transcription.implementation})</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Modelle</span>
                    <span class="info-value">${info.technology.transcription.models.join(', ')}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Sprechererkennung</span>
                    <span class="info-value">${info.technology.diarization.name}</span>
                </div>
                ${info.technology.glossary ? `
                <div class="info-row">
                    <span class="info-label">Begriffe (NER)</span>
                    <span class="info-value">${info.technology.glossary.name}</span>
                </div>
                ` : ''}
                <div class="info-row">
                    <span class="info-label">Backend</span>
                    <span class="info-value">${info.technology.backend.framework}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Laufzeit</span>
                    <span class="info-value">${info.technology.backend.language}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Frontend</span>
                    <span class="info-value">${info.technology.frontend.native_wrapper} + ${info.technology.frontend.type}</span>
                </div>
            </div>
        </div>

        <div class="info-section">
            <h3>Systemanforderungen</h3>
            <div class="info-grid">
                <div class="info-row">
                    <span class="info-label">Prozessor</span>
                    <span class="info-value">${info.system.requirements.processor}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">RAM</span>
                    <span class="info-value">${info.system.requirements.ram}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Speicher</span>
                    <span class="info-value">${info.system.requirements.storage}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">macOS</span>
                    <span class="info-value">${info.system.requirements.macos}</span>
                </div>
            </div>
        </div>

        <div class="info-section">
            <h3>Datenschutz & DSGVO</h3>
            <div class="info-grid">
                <div class="info-row">
                    <span class="info-label">Datenstandort</span>
                    <span class="info-value success">${info.privacy.data_location}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Cloud</span>
                    <span class="info-value success">${info.privacy.cloud_connection}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Uploads</span>
                    <span class="info-value">${info.privacy.auto_cleanup}</span>
                </div>
            </div>
            <div class="privacy-note">
                <p><strong>DSGVO-konform:</strong> ${info.privacy.dsgvo.reason}</p>
            </div>
        </div>

        <div class="info-section">
            <h3>Lizenzen</h3>
            <div class="info-grid">
                <div class="info-row">
                    <span class="info-label">LocalTranscript</span>
                    <span class="info-value"><span class="info-badge">${info.app.license}</span></span>
                </div>
                <div class="info-row">
                    <span class="info-label">Whisper</span>
                    <span class="info-value"><span class="info-badge">${info.technology.transcription.license}</span></span>
                </div>
                <div class="info-row">
                    <span class="info-label">${info.technology.diarization.name}</span>
                    <span class="info-value"><span class="info-badge">${info.technology.diarization.license}</span></span>
                </div>
                ${info.technology.glossary ? `
                <div class="info-row">
                    <span class="info-label">${info.technology.glossary.name}</span>
                    <span class="info-value"><span class="info-badge">${info.technology.glossary.license}</span></span>
                </div>
                ` : ''}
                <div class="info-row">
                    <span class="info-label">FastAPI / Electron</span>
                    <span class="info-value"><span class="info-badge">${info.technology.backend.license}</span></span>
                </div>
                <div class="info-row">
                    <span class="info-label">Drittanbieter</span>
                    <span class="info-value">
                        <a href="#" id="infoOpenLicenses">Vollständige Liste &amp; Quellen anzeigen</a>
                    </span>
                </div>
            </div>
        </div>
    `;

    infoContent.innerHTML = html;

    const openFolderLink = document.getElementById('infoOpenFolder');
    if (openFolderLink && isElectron && config.transcriptsRoot) {
        openFolderLink.addEventListener('click', (e) => {
            e.preventDefault();
            electronBridge.openFolder(config.transcriptsRoot).catch(err => {
                alert('Konnte Ordner nicht öffnen: ' + (err.message || err));
            });
        });
    }

    const openRepoLink = document.getElementById('infoOpenRepo');
    if (openRepoLink) {
        openRepoLink.addEventListener('click', (e) => {
            e.preventDefault();
            const url = info.app.repository;
            if (isElectron && electronBridge.openExternal) {
                electronBridge.openExternal(url).catch(() => window.open(url, '_blank'));
            } else {
                window.open(url, '_blank');
            }
        });
    }

    const openLicensesLink = document.getElementById('infoOpenLicenses');
    if (openLicensesLink) {
        openLicensesLink.addEventListener('click', (e) => {
            e.preventDefault();
            if (isElectron && electronBridge.openLicenses) {
                electronBridge.openLicenses().catch(err => {
                    alert('Konnte Lizenzdatei nicht öffnen: ' + (err.message || err));
                });
            } else {
                window.open('https://github.com/BenPohlBasel/LocalTranscript/blob/main/THIRD-PARTY-LICENSES.md', '_blank');
            }
        });
    }
}
