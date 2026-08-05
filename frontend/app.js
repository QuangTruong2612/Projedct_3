/* ==========================================================================
   RECRUITMENT AI DASHBOARD - APPLICATION JAVASCRIPT
   ========================================================================== */

// API Configuration
const API_BASE_URL = "http://localhost:8000";

// Global State
let selectedFiles = [];
let selectedJdFile = null;
let rankedResults = [];
let isApiOnline = false;

// DOM Elements
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  checkApiHealth();
  setupDropzone();
  setupJdFileUpload();
  setupEventListeners();
});

/* ==========================================================================
   THEME TOGGLE
   ========================================================================== */
function initTheme() {
  const savedTheme = localStorage.getItem("recruitment_ai_theme") || "dark";
  document.documentElement.setAttribute("data-theme", savedTheme);
  updateThemeIcon(savedTheme);

  document.getElementById("themeToggle").addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("recruitment_ai_theme", next);
    updateThemeIcon(next);
  });
}

function updateThemeIcon(theme) {
  const icon = document.querySelector("#themeToggle i");
  if (theme === "dark") {
    icon.className = "fa-solid fa-sun";
  } else {
    icon.className = "fa-solid fa-moon";
  }
}

/* ==========================================================================
   API HEALTH CHECK
   ========================================================================== */
async function checkApiHealth() {
  const badge = document.getElementById("apiStatusBadge");
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      isApiOnline = true;
      badge.innerHTML = `
        <span class="status-dot green"></span>
        <span class="status-text">Backend API (Live)</span>
      `;
      return;
    }
  } catch (err) {
    // API Offline
  }

  isApiOnline = false;
  badge.innerHTML = `
    <span class="status-dot amber"></span>
    <span class="status-text">Chế độ Demo (Offline)</span>
  `;
}



/* ==========================================================================
   DROPZONE & FILE UPLOAD
   ========================================================================== */
function setupDropzone() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("cvFilesInput");

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFilesSelected(Array.from(e.dataTransfer.files));
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFilesSelected(Array.from(e.target.files));
    }
  });
}

function handleFilesSelected(newFiles) {
  // Append new unique files
  newFiles.forEach(file => {
    if (!selectedFiles.some(f => f.name === file.name && f.size === file.size)) {
      selectedFiles.push(file);
    }
  });
  renderFilesList();
}

function removeFile(index) {
  selectedFiles.splice(index, 1);
  renderFilesList();
}

function renderFilesList() {
  const listEl = document.getElementById("filesList");
  const badgeEl = document.getElementById("fileCountBadge");

  badgeEl.textContent = `${selectedFiles.length} file`;

  if (selectedFiles.length === 0) {
    listEl.innerHTML = "";
    return;
  }

  listEl.innerHTML = selectedFiles.map((file, idx) => `
    <div class="file-item-card">
      <div class="file-info">
        <i class="${getFileIcon(file.name)}"></i>
        <span class="file-name text-ellipsis" title="${file.name}">${file.name}</span>
        <span class="file-size">(${formatBytes(file.size)})</span>
      </div>
      <button class="file-remove-btn" onclick="removeFile(${idx})" title="Xóa file">
        <i class="fa-solid fa-trash-can"></i>
      </button>
    </div>
  `).join("");
}

function getFileIcon(filename) {
  const ext = filename.split(".").pop().toLowerCase();
  if (ext === "pdf") return "fa-solid fa-file-pdf";
  if (ext === "docx" || ext === "doc") return "fa-solid fa-file-word";
  return "fa-solid fa-file-lines";
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

/* ==========================================================================
   FORM SUBMISSION & AI EVALUATION
   ========================================================================== */
function setupEventListeners() {
  document.getElementById("btnSubmitRank").addEventListener("click", processRankForm);
  document.getElementById("btnDemoData").addEventListener("click", loadDemoData);
}

/* ==========================================================================
   JD FILE UPLOAD DROPZONE HANDLERS
   ========================================================================== */
function setupJdFileUpload() {
  const dropzone = document.getElementById("jdDropzone");
  const fileInput = document.getElementById("jdFileInput");

  if (!dropzone || !fileInput) return;

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleJdFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleJdFileSelected(e.target.files[0]);
    }
  });
}

function handleJdFileSelected(file) {
  selectedJdFile = file;
  renderJdFileContainer();

  // If text file, auto fill textarea
  const ext = file.name.split('.').pop().toLowerCase();
  if (ext === 'txt') {
    const reader = new FileReader();
    reader.onload = (e) => {
      document.getElementById("jdText").value = e.target.result;
    };
    reader.readAsText(file);
  }
  showToast(`Đã chọn file JD: ${file.name}`, "info");
}

function clearJdFile() {
  selectedJdFile = null;
  const input = document.getElementById("jdFileInput");
  if (input) input.value = "";
  renderJdFileContainer();
}

function renderJdFileContainer() {
  const container = document.getElementById("jdFileContainer");
  const badge = document.getElementById("jdStatusBadge");
  if (!container) return;

  if (!selectedJdFile) {
    container.innerHTML = "";
    if (badge) {
      badge.textContent = "Chưa chọn file";
      badge.className = "badge badge-info";
    }
    return;
  }

  if (badge) {
    badge.textContent = "Đã chọn 1 file JD";
    badge.className = "badge badge-info green";
  }

  container.innerHTML = `
    <div class="file-item-card">
      <div class="file-info">
        <i class="${getFileIcon(selectedJdFile.name)}"></i>
        <span class="file-name text-ellipsis" title="${selectedJdFile.name}">File JD: ${selectedJdFile.name}</span>
        <span class="file-size">(${formatBytes(selectedJdFile.size)})</span>
      </div>
      <button type="button" class="file-remove-btn" onclick="clearJdFile()" title="Xóa file JD">
        <i class="fa-solid fa-trash-can"></i>
      </button>
    </div>
  `;
}

async function processRankForm() {
  const jdId = document.getElementById("jdId").value.trim();
  const jdText = document.getElementById("jdText").value.trim();

  if (!jdText && !selectedJdFile) {
    showToast("Vui lòng nhập nội dung JD hoặc tải lên file JD!", "error");
    return;
  }

  if (selectedFiles.length === 0) {
    showToast("Vui lòng tải lên ít nhất 1 file CV để đánh giá!", "error");
    return;
  }

  showLoadingState();

  if (isApiOnline) {
    try {
      // Step 1 animation
      await setStep(1);
      
      const formData = new FormData();
      selectedFiles.forEach(file => {
        formData.append("cv_files", file);
      });
      if (selectedJdFile) {
        formData.append("jd_file", selectedJdFile);
      }
      if (jdText) {
        formData.append("jd_text", jdText);
      }
      formData.append("jd_id", jdId || "JD-01");

      await setStep(2);

      const response = await fetch(`${API_BASE_URL}/api/v1/rank`, {
        method: "POST",
        body: formData
      });

      await setStep(3);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Lỗi gọi API backend");
      }

      const data = await response.json();
      rankedResults = data.results || [];
      renderResults(rankedResults);
      showToast(`Đã phân tích xong ${rankedResults.length} hồ sơ!`, "success");
    } catch (err) {
      console.warn("API Error, fallback to AI simulation:", err);
      showToast(`Backend API: ${err.message}. Tự động mô phỏng đánh giá!`, "info");
      await simulateAiEvaluation(selectedFiles, jdId);
    }
  } else {
    // Offline simulation
    await simulateAiEvaluation(selectedFiles, jdId);
  }

  hideLoadingState();
}

/* Simulate AI Evaluation when offline or testing */
async function simulateAiEvaluation(files, jdId) {
  await setStep(1);
  await delay(600);
  await setStep(2);
  await delay(800);
  await setStep(3);
  await delay(600);

  const mockNames = [
    "Nguyen_Van_An_CV.pdf",
    "Tran_Thi_Binh_Resume.docx",
    "Le_Hoang_Cường_CV.pdf",
    "Pham_Duc_Duy_Fullstack.pdf",
    "Vo_Minh_E_AI_Engineer.docx"
  ];

  rankedResults = files.map((file, i) => {
    // Random high scores for realistic UI display
    const finalScore = Math.floor(Math.random() * 35) + 65; // 65 to 99
    const ruleScore = Math.min(100, Math.floor(finalScore + (Math.random() * 10 - 5)));

    return {
      rank: i + 1,
      rule_based_score: ruleScore,
      evaluation: {
        cv_id: file.name,
        jd_id: jdId,
        final_score: finalScore,
        strengths: [
          "Có trên 3 năm kinh nghiệm lập trình chuyên sâu",
          "Thành thạo các framework yêu cầu trong JD",
          "Kinh nghiệm làm dự án thực tế quy mô lớn",
          "Kỹ năng làm việc nhóm và tư duy giải quyết vấn đề tốt"
        ].slice(0, Math.floor(Math.random() * 2) + 2),
        gaps: [
          "Thiếu chứng chỉ quốc tế liên quan",
          "Kinh nghiệm với Docker/K8s còn hạn chế",
          "Chưa đề cập mức lương kỳ vọng"
        ].slice(0, Math.floor(Math.random() * 2) + 1),
        explanation: `Ứng viên thể hiện năng lực chuyên môn rất tốt với tổng điểm AI đạt ${finalScore}/100. Kỹ năng cốt lõi hoàn toàn phù hợp với các yêu cầu bắt buộc trong Job Description (${jdId}). Khuyến nghị mời phỏng vấn vòng 1.`
      }
    };
  });

  // Sort by final score descending
  rankedResults.sort((a, b) => b.evaluation.final_score - a.evaluation.final_score);
  rankedResults.forEach((r, idx) => r.rank = idx + 1);

  renderResults(rankedResults);
  showToast(`Phân tích thành công ${rankedResults.length} hồ sơ ứng viên!`, "success");
}

async function loadDemoData() {
  const dummyJdFile = new File([
    "Vị trí: Senior AI / Machine Learning Engineer\nYêu cầu:\n- Tối thiểu 3 năm kinh nghiệm AI/ML, Python, PyTorch/TensorFlow, LLMs, RAG.\n- Bằng cử nhân CNTT hoặc tương đương."
  ], "JD_Senior_AI_Engineer.pdf", { type: "application/pdf" });

  const dummyFiles = [
    new File(["Sample content 1"], "Nguyen_Van_Hung_AI_Engineer.pdf", { type: "application/pdf" }),
    new File(["Sample content 2"], "Tran_Le_Minh_Senior_Dev.docx", { type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document" }),
    new File(["Sample content 3"], "Pham_Quoc_Anh_CV.pdf", { type: "application/pdf" })
  ];

  selectedJdFile = dummyJdFile;
  renderJdFileContainer();

  selectedFiles = dummyFiles;
  renderFilesList();

  document.getElementById("jdId").value = "JD-AI-ENG-01";
  processRankForm();
}

/* Loading Animations */
function showLoadingState() {
  document.getElementById("emptyState").classList.add("hidden");
  document.getElementById("resultsContent").classList.add("hidden");
  document.getElementById("loadingState").classList.remove("hidden");
}

function hideLoadingState() {
  document.getElementById("loadingState").classList.add("hidden");
  document.getElementById("resultsContent").classList.remove("hidden");
}

async function setStep(stepNum) {
  document.querySelectorAll(".step-item").forEach((el, idx) => {
    if (idx + 1 <= stepNum) el.classList.add("active");
    else el.classList.remove("active");
  });
  
  if (stepNum >= 2) document.getElementById("line1").classList.add("active");
  if (stepNum >= 3) document.getElementById("line2").classList.add("active");
}

function delay(ms) { return new Promise(res => setTimeout(res, ms)); }

/* ==========================================================================
   RENDER RESULTS & METRICS
   ========================================================================== */
function renderResults(results) {
  if (!results || results.length === 0) return;

  // 1. Calculate Metrics
  const total = results.length;
  const highest = Math.max(...results.map(r => r.evaluation.final_score));
  const avg = Math.round(results.reduce((acc, r) => acc + r.evaluation.final_score, 0) / total);
  const topCandidate = results[0].evaluation.cv_id;

  document.getElementById("metricTotal").textContent = total;
  document.getElementById("metricHighest").textContent = `${highest}/100`;
  document.getElementById("metricAvg").textContent = `${avg}/100`;
  document.getElementById("metricTopName").textContent = topCandidate;

  // 2. Render Candidate Cards List
  filterCandidates();
}

function filterCandidates() {
  const searchTerm = document.getElementById("searchCandidate").value.toLowerCase();
  const filterTier = document.getElementById("filterScore").value;

  const filtered = rankedResults.filter(r => {
    const nameMatch = r.evaluation.cv_id.toLowerCase().includes(searchTerm);
    const score = r.evaluation.final_score;

    let scoreMatch = true;
    if (filterTier === "high") scoreMatch = score >= 80;
    else if (filterTier === "medium") scoreMatch = score >= 50 && score < 80;
    else if (filterTier === "low") scoreMatch = score < 50;

    return nameMatch && scoreMatch;
  });

  document.getElementById("resultsCountText").textContent = `Hiển thị ${filtered.length} / ${rankedResults.length} kết quả`;

  const container = document.getElementById("candidatesList");

  if (filtered.length === 0) {
    container.innerHTML = `
      <div style="text-align: center; padding: 2rem; color: var(--text-muted);">
        <i class="fa-solid fa-filter-circle-xmark" style="font-size: 2rem; margin-bottom: 0.5rem;"></i>
        <p>Không tìm thấy ứng viên phù hợp với bộ lọc</p>
      </div>
    `;
    return;
  }

  container.innerHTML = filtered.map(r => {
    const ev = r.evaluation;
    const scoreClass = ev.final_score >= 80 ? 'high' : ev.final_score >= 50 ? 'medium' : 'low';
    const rankClass = r.rank <= 3 ? `rank-${r.rank}` : '';

    return `
      <div class="candidate-card">
        <div class="candidate-left">
          <div class="rank-badge ${rankClass}">
            ${r.rank <= 3 ? `<i class="fa-solid fa-trophy"></i>` : `#${r.rank}`}
          </div>
          <div class="candidate-info">
            <h4 class="text-ellipsis" title="${ev.cv_id}">${ev.cv_id}</h4>
            <div class="candidate-summary">
              <span class="pill-tag green"><i class="fa-solid fa-check"></i> ${ev.strengths.length} điểm mạnh</span>
              <span class="pill-tag orange"><i class="fa-solid fa-circle-info"></i> ${ev.gaps.length} điểm thiếu</span>
            </div>
          </div>
        </div>

        <div class="candidate-right">
          <div class="score-badge-box">
            <div class="score-number ${scoreClass}">${ev.final_score}</div>
            <div class="score-label">AI Match Score</div>
          </div>

          <button class="btn btn-secondary btn-sm" onclick="openDetailModal(${r.rank - 1})">
            <i class="fa-solid fa-eye"></i> Chi tiết
          </button>
        </div>
      </div>
    `;
  }).join("");
}

/* ==========================================================================
   CANDIDATE DETAIL MODAL
   ========================================================================== */
function openDetailModal(index) {
  const item = rankedResults[index];
  if (!item) return;

  const ev = item.evaluation;

  document.getElementById("modalRank").textContent = `#${item.rank}`;
  document.getElementById("modalCvName").textContent = ev.cv_id;
  document.getElementById("modalJdId").textContent = `Mã vị trí: ${ev.jd_id}`;

  document.getElementById("modalFinalScore").innerHTML = `${ev.final_score}<span>/100</span>`;
  document.getElementById("modalRuleScore").innerHTML = `${item.rule_based_score}<span>/100</span>`;

  // Strengths
  const strengthsEl = document.getElementById("modalStrengths");
  strengthsEl.innerHTML = ev.strengths.map(s => `
    <div class="tag-item"><i class="fa-solid fa-check"></i> ${s}</div>
  `).join("") || '<p class="text-sub">Chưa có thông tin điểm mạnh</p>';

  // Gaps
  const gapsEl = document.getElementById("modalGaps");
  gapsEl.innerHTML = ev.gaps.map(g => `
    <div class="tag-item"><i class="fa-solid fa-exclamation-triangle"></i> ${g}</div>
  `).join("") || '<p class="text-sub">Không có điểm thiếu sót đáng kể</p>';

  // Explanation
  document.getElementById("modalExplanation").textContent = ev.explanation || "Không có nhận xét chi tiết.";

  document.getElementById("detailModal").classList.remove("hidden");
}

function closeDetailModal() {
  document.getElementById("detailModal").classList.add("hidden");
}

/* Close modal on ESC key or clicking background */
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeDetailModal();
});

document.getElementById("detailModal").addEventListener("click", (e) => {
  if (e.target.id === "detailModal") closeDetailModal();
});

/* ==========================================================================
   EXPORT RESULTS TO CSV
   ========================================================================== */
function exportResultsCSV() {
  if (rankedResults.length === 0) {
    showToast("Không có dữ liệu để xuất file!", "error");
    return;
  }

  let csvContent = "data:text/csv;charset=utf-8,\uFEFF";
  csvContent += "Hạng,Tên CV,Mã JD,Điểm AI Score,Điểm Rule Match,Điểm Mạnh,Điểm Thiếu Hụt\n";

  rankedResults.forEach(r => {
    const ev = r.evaluation;
    const strengthsStr = `"${ev.strengths.join('; ')}"`;
    const gapsStr = `"${ev.gaps.join('; ')}"`;
    csvContent += `${r.rank},"${ev.cv_id}","${ev.jd_id}",${ev.final_score},${r.rule_based_score},${strengthsStr},${gapsStr}\n`;
  });

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `Recruitment_AI_Report_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  showToast("Đã xuất báo cáo CSV thành công!", "success");
}

/* ==========================================================================
   TOAST NOTIFICATION SYSTEM
   ========================================================================== */
function showToast(message, type = "info") {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const iconMap = {
    success: "fa-solid fa-circle-check text-success",
    error: "fa-solid fa-circle-xmark text-danger",
    info: "fa-solid fa-circle-info text-primary"
  };

  toast.innerHTML = `
    <i class="${iconMap[type] || iconMap.info}"></i>
    <span>${message}</span>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(100%)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}
