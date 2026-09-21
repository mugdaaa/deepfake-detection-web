// VeritasAI // Deepfake Detection Lab Client Application

let currentSelectedFile = null;
let currentSampleId = null;
let benchmarkChart = null;
let lastAnalysisResult = null;

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Lucide icons
  lucide.createIcons();

  // Setup Drag & Drop
  setupDropZone();

  // Setup Sliders
  setupSliders();

  // Setup Analyze Button
  document.getElementById('btn-analyze').addEventListener('click', handleAnalyze);

  // Setup Remove Media
  document.getElementById('btn-remove-media').addEventListener('click', clearMediaInput);

  // Fetch benchmark data & build chart
  initBenchmarkChart();
});

// Tab Switching
function switchTab(tabId) {
  const tabs = ['studio', 'explainability', 'benchmarks', 'about'];
  tabs.forEach(t => {
    const pane = document.getElementById(`tab-${t}`);
    const btn = document.getElementById(`tab-btn-${t}`);
    if (t === tabId) {
      pane.classList.remove('hidden');
      btn.classList.add('active');
      btn.classList.remove('text-slate-400');
    } else {
      pane.classList.add('hidden');
      btn.classList.remove('active');
      btn.classList.add('text-slate-400');
    }
  });

  if (tabId === 'benchmarks' && benchmarkChart) {
    benchmarkChart.resize();
  }

  // Refresh icons
  lucide.createIcons();
}

// Setup Drag and Drop File Zone
function setupDropZone() {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');

  dropZone.addEventListener('click', (e) => {
    if (e.target.closest('#btn-remove-media')) return;
    fileInput.click();
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.add('border-sky-400', 'bg-cyber-850');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.remove('border-sky-400', 'bg-cyber-850');
    }, false);
  });

  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (fileInput.files && fileInput.files.length > 0) {
      handleFileSelected(fileInput.files[0]);
    }
  });
}

function handleFileSelected(file) {
  currentSelectedFile = file;
  currentSampleId = null;

  const dropPrompt = document.getElementById('drop-prompt');
  const mediaPreview = document.getElementById('media-preview');
  const previewVideo = document.getElementById('preview-video');
  const previewImage = document.getElementById('preview-image');
  const previewFilename = document.getElementById('preview-filename');
  const previewFilesize = document.getElementById('preview-filesize');

  previewFilename.textContent = file.name;
  previewFilesize.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';

  const fileUrl = URL.createObjectURL(file);
  const isVideo = file.type.startsWith('video/') || file.name.match(/\.(mp4|avi|mov|webm)$/i);

  if (isVideo) {
    previewImage.classList.add('hidden');
    previewVideo.classList.remove('hidden');
    previewVideo.src = fileUrl;
    previewVideo.load();
  } else {
    previewVideo.classList.add('hidden');
    previewImage.classList.remove('hidden');
    previewImage.src = fileUrl;
  }

  dropPrompt.classList.add('hidden');
  mediaPreview.classList.remove('hidden');
  lucide.createIcons();
}

function clearMediaInput(e) {
  if (e) e.stopPropagation();
  currentSelectedFile = null;
  currentSampleId = null;

  document.getElementById('file-input').value = '';
  document.getElementById('drop-prompt').classList.remove('hidden');
  document.getElementById('media-preview').classList.add('hidden');
  document.getElementById('preview-video').src = '';
  document.getElementById('preview-image').src = '';
}

// Sliders Event Setup
function setupSliders() {
  const framesSlider = document.getElementById('num-frames-slider');
  const framesVal = document.getElementById('num-frames-val');
  framesSlider.addEventListener('input', (e) => {
    framesVal.textContent = `${e.target.value} Frames`;
  });

  const threshSlider = document.getElementById('threshold-slider');
  const threshVal = document.getElementById('threshold-val');
  threshSlider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value).toFixed(2);
    let note = '';
    if (val <= 0.35) note = ' (Celeb-DF Generalization)';
    else if (val == 0.50) note = ' (Standard)';
    else if (val >= 0.65) note = ' (Conservative)';
    threshVal.textContent = `${val}${note}`;
  });
}

// 1-Click Sample Loading
function loadSample(sampleId) {
  currentSampleId = sampleId;
  currentSelectedFile = null;

  const dropPrompt = document.getElementById('drop-prompt');
  const mediaPreview = document.getElementById('media-preview');
  const previewVideo = document.getElementById('preview-video');
  const previewImage = document.getElementById('preview-image');
  const previewFilename = document.getElementById('preview-filename');
  const previewFilesize = document.getElementById('preview-filesize');

  let filename = '';
  let isVideo = false;

  if (sampleId === 'sample_fake_video') {
    filename = 'gradcam_fake_output.mp4 (Manipulated FaceForensics++ c23)';
    isVideo = true;
    previewVideo.src = '/assets/sample_fake_video';
  } else if (sampleId === 'sample_real_video') {
    filename = 'gradcam_real_output.mp4 (Authentic Sequence)';
    isVideo = true;
    previewVideo.src = '/assets/sample_real_video';
  } else if (sampleId === 'face_extraction_sample') {
    filename = 'face_extraction_sample.png (Extracted Aligned Faces)';
    previewImage.src = '/assets/face_extraction_sample';
  } else if (sampleId === 'background_masking_sample') {
    filename = 'background_masking_sample.png (Elliptical Mask Sample)';
    previewImage.src = '/assets/background_masking_sample';
  }

  previewFilename.textContent = filename;
  previewFilesize.textContent = 'Preloaded Sample';

  if (isVideo) {
    previewImage.classList.add('hidden');
    previewVideo.classList.remove('hidden');
    previewVideo.load();
  } else {
    previewVideo.classList.add('hidden');
    previewImage.classList.remove('hidden');
  }

  dropPrompt.classList.add('hidden');
  mediaPreview.classList.remove('hidden');
  lucide.createIcons();

  // Trigger analysis automatically
  handleAnalyze();
}

// Handle Analysis Trigger
async function handleAnalyze() {
  if (!currentSelectedFile && !currentSampleId) {
    alert('Please upload a video/image or select a quick test sample.');
    return;
  }

  const numFrames = document.getElementById('num-frames-slider').value;
  const applyMask = document.getElementById('apply-mask-toggle').checked;
  const generateHeatmap = document.getElementById('gradcam-toggle').checked;
  const threshold = document.getElementById('threshold-slider').value;

  // Toggle UI state to loading
  document.getElementById('results-empty').classList.add('hidden');
  document.getElementById('results-content').classList.add('hidden');
  document.getElementById('results-loading').classList.remove('hidden');

  const btnAnalyze = document.getElementById('btn-analyze');
  btnAnalyze.disabled = true;
  document.getElementById('btn-analyze-text').textContent = 'Analyzing Neural Features...';

  try {
    let response;
    if (currentSelectedFile) {
      const formData = new FormData();
      formData.append('file', currentSelectedFile);
      formData.append('num_frames', numFrames);
      formData.append('apply_mask', applyMask);
      formData.append('generate_heatmap', generateHeatmap);
      formData.append('threshold', threshold);

      response = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });
    } else {
      const formData = new FormData();
      formData.append('sample_id', currentSampleId);
      formData.append('num_frames', numFrames);
      formData.append('apply_mask', applyMask);
      formData.append('generate_heatmap', generateHeatmap);
      formData.append('threshold', threshold);

      response = await fetch('/api/predict-sample', {
        method: 'POST',
        body: formData
      });
    }

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Inference failed');
    }

    const result = await response.json();
    lastAnalysisResult = result;
    renderResults(result);

  } catch (error) {
    alert(`Error during analysis: ${error.message}`);
    document.getElementById('results-empty').classList.remove('hidden');
  } finally {
    document.getElementById('results-loading').classList.add('hidden');
    btnAnalyze.disabled = false;
    document.getElementById('btn-analyze-text').textContent = 'Analyze Media for Deepfakes';
  }
}

// Render Results in UI
function renderResults(res) {
  const container = document.getElementById('results-content');
  container.classList.remove('hidden');

  const banner = document.getElementById('verdict-banner');
  const verdictTitle = document.getElementById('verdict-title');
  const verdictConfidence = document.getElementById('verdict-confidence');
  const verdictExpl = document.getElementById('verdict-explanation');
  const verdictProbVal = document.getElementById('verdict-prob-value');
  const verdictProbBar = document.getElementById('verdict-prob-bar');
  const statusIcon = document.getElementById('verdict-status-icon');

  const isFake = res.verdict === 'FAKE';
  const isInconclusive = res.verdict === 'INCONCLUSIVE';

  // Apply colors and styling based on verdict
  if (isInconclusive) {
    banner.className = 'rounded-2xl p-6 border transition-all relative overflow-hidden bg-amber-950/40 border-amber-500/40 text-amber-300';
    verdictTitle.textContent = 'INCONCLUSIVE';
    verdictConfidence.textContent = '0%';
    verdictExpl.textContent = res.message || 'No clear human faces detected in the provided media.';
    verdictProbVal.textContent = 'N/A';
    verdictProbBar.style.width = '0%';
    verdictProbBar.className = 'h-full rounded-full bg-amber-500';
    statusIcon.innerHTML = '<i data-lucide="help-circle" class="w-8 h-8 text-amber-400"></i>';
  } else if (isFake) {
    banner.className = 'rounded-2xl p-6 border transition-all relative overflow-hidden bg-rose-950/40 border-rose-500/50 text-rose-300 shadow-lg shadow-rose-950/50';
    verdictTitle.textContent = 'MANIPULATED (FAKE)';
    verdictConfidence.textContent = `${res.confidence.toFixed(1)}%`;
    verdictExpl.textContent = `High confidence facial boundary & synthesis artifacts detected by spatial attention module (Probability: ${res.fake_probability_percent}%).`;
    verdictProbVal.textContent = `${(res.fake_probability * 100).toFixed(2)}%`;
    verdictProbBar.style.width = `${Math.min(100, res.fake_probability * 100)}%`;
    verdictProbBar.className = 'h-full rounded-full bg-gradient-to-r from-amber-500 to-rose-500 shadow-md shadow-rose-500/40';
    statusIcon.innerHTML = '<i data-lucide="alert-triangle" class="w-8 h-8 text-rose-400 animate-pulse"></i>';
  } else {
    banner.className = 'rounded-2xl p-6 border transition-all relative overflow-hidden bg-emerald-950/40 border-emerald-500/50 text-emerald-300 shadow-lg shadow-emerald-950/50';
    verdictTitle.textContent = 'AUTHENTIC (REAL)';
    verdictConfidence.textContent = `${res.confidence.toFixed(1)}%`;
    verdictExpl.textContent = `No significant manipulation or warping anomalies found. Natural skin and frequency textures intact (Fake Likelihood: ${res.fake_probability_percent}%).`;
    verdictProbVal.textContent = `${(res.fake_probability * 100).toFixed(2)}%`;
    verdictProbBar.style.width = `${Math.min(100, res.fake_probability * 100)}%`;
    verdictProbBar.className = 'h-full rounded-full bg-gradient-to-r from-sky-500 to-emerald-400 shadow-md shadow-emerald-500/40';
    statusIcon.innerHTML = '<i data-lucide="shield-check" class="w-8 h-8 text-emerald-400"></i>';
  }

  // Update KPI metric tiles
  document.getElementById('metric-faces-count').textContent = res.media_type === 'video' ? (res.frames_with_faces > 0 ? 'Detected' : '0') : res.faces_detected;
  document.getElementById('metric-frames-count').textContent = res.media_type === 'video' ? res.frames_with_faces : 1;
  document.getElementById('metric-media-info').textContent = res.media_type === 'video' ? `${res.duration_seconds}s` : `${res.dimensions?.width}x${res.dimensions?.height}`;
  document.getElementById('metric-threshold').textContent = res.threshold_used;

  // Build Frames Carousel
  const carousel = document.getElementById('frames-carousel');
  carousel.innerHTML = '';

  const frames = res.media_type === 'video' ? res.frames_data : res.analyzed_faces;

  if (frames && frames.length > 0) {
    frames.forEach((frame, idx) => {
      const card = document.createElement('div');
      card.className = `frame-card cursor-pointer flex-shrink-0 w-28 p-2 rounded-xl bg-cyber-900 border border-cyber-700/80 text-center space-y-1.5 ${idx === 0 ? 'active-frame' : ''}`;
      
      const thumb = frame.heatmap_url || frame.face_url;
      const prob = frame.fake_prob_percent;
      const badgeColor = prob > 50 ? 'bg-rose-500/20 text-rose-400 border-rose-500/30' : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';

      card.innerHTML = `
        <div class="aspect-square rounded-lg overflow-hidden bg-black flex items-center justify-center">
          <img src="${thumb}" alt="Frame ${idx+1}" class="w-full h-full object-cover">
        </div>
        <div class="text-[10px] font-mono text-slate-400">${res.media_type === 'video' ? `Frame #${frame.frame_index}` : `Face #${frame.face_index}`}</div>
        <div class="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border ${badgeColor}">${prob}% Fake</div>
      `;

      card.addEventListener('click', () => {
        document.querySelectorAll('.frame-card').forEach(c => c.classList.remove('active-frame'));
        card.classList.add('active-frame');
        showFrameDetail(frame, idx, res.media_type);
      });

      carousel.appendChild(card);
    });

    // Initialize inspector with first frame
    showFrameDetail(frames[0], 0, res.media_type);
  }

  lucide.createIcons();
}

// Display Selected Frame in Inspector
function showFrameDetail(frame, idx, mediaType) {
  const badge = document.getElementById('inspector-badge');
  badge.textContent = mediaType === 'video' ? `Frame #${frame.frame_index} (${frame.timestamp_sec}s)` : `Face #${frame.face_index}`;

  const originalImg = document.getElementById('inspector-original-img');
  const heatmapImg = document.getElementById('inspector-heatmap-img');
  const probSpan = document.getElementById('inspector-frame-prob');

  originalImg.src = frame.face_url;
  heatmapImg.src = frame.heatmap_url || frame.face_url;
  
  const prob = frame.fake_prob_percent;
  probSpan.textContent = `${prob}% Fake Likelihood`;
  probSpan.className = `font-bold ${prob > 50 ? 'text-rose-400' : 'text-emerald-400'}`;
}

// Confusion Matrix Tab Switcher
function showConfusionMatrix(modelKey) {
  const tabs = ['ensemble', 'xception', 'efficientnet', 'rf', 'svm'];
  tabs.forEach(t => {
    const btn = document.getElementById(`cm-btn-${t}`);
    if (t === modelKey) {
      btn.classList.add('active');
      btn.classList.remove('text-slate-400');
    } else {
      btn.classList.remove('active');
      btn.classList.add('text-slate-400');
    }
  });

  const img = document.getElementById('cm-display-img');
  img.src = `/assets/${modelKey}_confusion_matrix`;
}

// Chart.js Setup for Research Benchmarks
async function initBenchmarkChart() {
  try {
    const res = await fetch('/api/metrics');
    const data = await res.json();

    const ctx = document.getElementById('modelComparisonChart').getContext('2d');
    const models = data.models;

    const labels = models.map(m => m.name);
    const accuracies = models.map(m => m.accuracy);
    const aucs = models.map(m => (m.auc * 100).toFixed(1));

    benchmarkChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Accuracy (%)',
            data: accuracies,
            backgroundColor: 'rgba(56, 189, 248, 0.85)',
            borderColor: '#38BDF8',
            borderWidth: 1,
            borderRadius: 6
          },
          {
            label: 'AUC (x100)',
            data: aucs,
            backgroundColor: 'rgba(99, 102, 241, 0.85)',
            borderColor: '#6366F1',
            borderWidth: 1,
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: {
              color: '#CBD5E1',
              font: { family: '"JetBrains Mono"', size: 11 }
            }
          },
          tooltip: {
            backgroundColor: '#0D1322',
            titleColor: '#F8FAFC',
            bodyColor: '#94A3B8',
            borderColor: '#1F315B',
            borderWidth: 1,
            padding: 10
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            grid: { color: 'rgba(31, 49, 91, 0.5)' },
            ticks: {
              color: '#94A3B8',
              font: { family: '"JetBrains Mono"', size: 10 }
            }
          },
          x: {
            grid: { display: false },
            ticks: {
              color: '#CBD5E1',
              font: { family: '"JetBrains Mono"', size: 10 }
            }
          }
        }
      }
    });
  } catch (err) {
    console.error('Failed to load benchmark metrics chart:', err);
  }
}
