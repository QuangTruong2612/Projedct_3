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
   ĐÁNH DẤU DỮ LIỆU MÔ PHỎNG
   ========================================================================== */
/** Bật/tắt dải cảnh báo trên bảng kết quả. Truyền null để ẩn. */
function setSimulatedBanner(reason) {
  const banner = document.getElementById("simulatedBanner");
  if (!banner) return;
  banner.classList.toggle("hidden", !reason);
  const detail = document.getElementById("simulatedReason");
  if (detail) detail.textContent = reason || "";
}

/** Backend đã pass health check nhưng lượt gọi thật bị lỗi -> badge không
    được phép tiếp tục hiển thị "Live". */
function markApiDegraded() {
  isApiOnline = false;
  const badge = document.getElementById("apiStatusBadge");
  if (!badge) return;
  badge.innerHTML = `
    <span class="status-dot amber"></span>
    <span class="status-text">Backend lỗi — đang mô phỏng</span>
  `;
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
      setSimulatedBanner(null);   // kết quả thật -> ẩn cảnh báo
      renderResults(rankedResults);
      showToast(`Đã phân tích xong ${rankedResults.length} hồ sơ!`, "success");
    } catch (err) {
      // Backend đã trả lời health check nhưng lượt chấm thật lại hỏng
      // (API key sai, 500, timeout...). Trước đây chỗ này lặng lẽ chuyển sang
      // dữ liệu giả trong khi badge vẫn báo "Live" -> người xem hoàn toàn có
      // thể tưởng số liệu bịa là kết quả thật.
      console.error("Lỗi gọi API /rank:", err);
      showToast(`Backend lỗi: ${err.message}`, "error");
      markApiDegraded();
      await simulateAiEvaluation(selectedFiles, jdId, `Backend lỗi: ${err.message}`);
    }
  } else {
    await simulateAiEvaluation(selectedFiles, jdId, "Backend API không kết nối được.");
  }

  hideLoadingState();
}

/* Simulate AI Evaluation when offline or testing.
   `reason` được hiển thị thường trực trên bảng kết quả để phân biệt rõ với
   kết quả chấm thật. */
async function simulateAiEvaluation(files, jdId, reason = "Đang chạy ở chế độ demo.") {
  setSimulatedBanner(reason);
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
   BẢNG PHÂN LOẠI (BOARD) CHO HR
   AI xếp sẵn ứng viên vào cột theo điểm; HR có quyền chuyển sang cột khác.
   Quyết định của HR được lưu lại và LUÔN ưu tiên hơn gợi ý của AI --
   máy đề xuất, người quyết định.
   ========================================================================== */

const BOARD_COLUMNS = [
  { id: "shortlist", title: "Nên phỏng vấn", icon: "fa-star", tone: "green" },
  { id: "consider",  title: "Cân nhắc thêm", icon: "fa-scale-balanced", tone: "amber" },
  { id: "rejected",  title: "Chưa phù hợp",  icon: "fa-circle-minus", tone: "red" },
];

let viewMode = "list";
/** cv_id -> id cột do HR tự chuyển. Chỉ chứa các ca HR ĐÃ can thiệp. */
let hrDecisions = {};

const DECISIONS_KEY = "recruitment_ai_hr_decisions";

function loadDecisions() {
  try {
    hrDecisions = JSON.parse(localStorage.getItem(DECISIONS_KEY)) || {};
  } catch (err) {
    hrDecisions = {};   // localStorage hỏng/bị chặn -> coi như chưa có quyết định nào
  }
}

function saveDecisions() {
  try {
    localStorage.setItem(DECISIONS_KEY, JSON.stringify(hrDecisions));
  } catch (err) {
    // Chế độ ẩn danh hoặc trình duyệt chặn -> vẫn chạy được, chỉ là không nhớ
    console.warn("Không lưu được quyết định vào localStorage:", err);
  }
}

/** Cột mà AI đề xuất, thuần theo điểm. */
function suggestedColumn(score) {
  if (score >= 80) return "shortlist";
  if (score >= 50) return "consider";
  return "rejected";
}

/** Cột thực tế đang hiển thị: ưu tiên quyết định của HR nếu có. */
function columnOf(item) {
  return hrDecisions[item.evaluation.cv_id] || suggestedColumn(item.evaluation.final_score);
}

function setViewMode(mode) {
  viewMode = mode;
  const isBoard = mode === "board";

  document.getElementById("candidatesList").classList.toggle("hidden", isBoard);
  document.getElementById("candidatesBoard").classList.toggle("hidden", !isBoard);

  const listBtn = document.getElementById("viewBtnList");
  const boardBtn = document.getElementById("viewBtnBoard");
  listBtn.classList.toggle("active", !isBoard);
  boardBtn.classList.toggle("active", isBoard);
  listBtn.setAttribute("aria-selected", String(!isBoard));
  boardBtn.setAttribute("aria-selected", String(isBoard));

  filterCandidates();
}

/** HR chuyển ứng viên sang cột khác. */
function moveCandidate(cvId, columnId) {
  const item = rankedResults.find(r => r.evaluation.cv_id === cvId);
  if (!item) return;

  if (columnId === suggestedColumn(item.evaluation.final_score)) {
    delete hrDecisions[cvId];   // trùng đề xuất AI -> không cần lưu đè
  } else {
    hrDecisions[cvId] = columnId;
  }
  saveDecisions();
  filterCandidates();

  const col = BOARD_COLUMNS.find(c => c.id === columnId);
  showToast(`Đã chuyển "${cvId}" sang "${col.title}"`, "success");
}

/** Thanh điểm nhỏ cho từng tiêu chí, dùng chung cho card và modal. */
function criterionBars(criterionScores) {
  if (!criterionScores || criterionScores.length === 0) return "";
  const label = { skills: "Kỹ năng", experience: "Kinh nghiệm", education: "Học vấn" };

  return criterionScores.map(c => {
    const pct = Math.round((c.score || 0) * 100);
    const tone = pct >= 70 ? "green" : pct >= 40 ? "amber" : "red";
    const warn = c.missing_data
      ? ` <i class="fa-solid fa-triangle-exclamation" title="Thiếu dữ liệu trong CV — điểm này không đáng tin"></i>`
      : "";
    return `
      <div class="criterion-bar" title="${escapeHtml(c.detail || "")}">
        <span class="criterion-name">${label[c.criterion] || c.criterion}${warn}</span>
        <div class="criterion-track"><div class="criterion-fill ${tone}" style="width:${pct}%"></div></div>
        <span class="criterion-value">${pct}</span>
      </div>`;
  }).join("");
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, ch => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]
  ));
}

function renderBoard(items) {
  const container = document.getElementById("candidatesBoard");

  container.innerHTML = BOARD_COLUMNS.map(col => {
    const inColumn = items.filter(r => columnOf(r) === col.id);

    const cards = inColumn.map(r => {
      const ev = r.evaluation;
      const scoreClass = ev.final_score >= 80 ? "high" : ev.final_score >= 50 ? "medium" : "low";
      const moved = Boolean(hrDecisions[ev.cv_id]);
      const unreliable = r.completeness && r.completeness.warnings && r.completeness.warnings.length > 0;

      const moveButtons = BOARD_COLUMNS
        .filter(c => c.id !== col.id)
        .map(c => `
          <button class="board-move-btn ${c.tone}" title="Chuyển sang ${c.title}"
                  onclick="moveCandidate('${escapeHtml(ev.cv_id)}', '${c.id}')">
            <i class="fa-solid ${c.icon}"></i>
          </button>`).join("");

      return `
        <article class="board-card">
          <header class="board-card-head">
            <span class="board-rank">#${r.rank}</span>
            <h5 class="text-ellipsis" title="${escapeHtml(ev.cv_id)}">${escapeHtml(ev.cv_id)}</h5>
            <span class="score-number ${scoreClass}">${ev.final_score}</span>
          </header>

          ${unreliable ? `
            <div class="board-card-warning" title="${escapeHtml(r.completeness.warnings.join("; "))}">
              <i class="fa-solid fa-triangle-exclamation"></i> CV thiếu dữ liệu — điểm chưa đáng tin
            </div>` : ""}

          <div class="board-card-criteria">${criterionBars(r.criterion_scores)}</div>

          <div class="board-card-tags">
            <span class="pill-tag green">${ev.strengths.length} mạnh</span>
            <span class="pill-tag orange">${ev.gaps.length} thiếu</span>
            ${moved ? `<span class="pill-tag blue" title="Bạn đã tự chuyển ứng viên này, khác với đề xuất của AI">HR đã đổi</span>` : ""}
          </div>

          <footer class="board-card-foot">
            <button class="btn btn-secondary btn-sm" onclick="openDetailModal(${r.rank - 1})">
              <i class="fa-solid fa-eye"></i> Chi tiết
            </button>
            <div class="board-move-group">${moveButtons}</div>
          </footer>
        </article>`;
    }).join("");

    return `
      <section class="board-column">
        <header class="board-column-head ${col.tone}">
          <h4><i class="fa-solid ${col.icon}"></i> ${col.title}</h4>
          <span class="board-count">${inColumn.length}</span>
        </header>
        <div class="board-column-body">
          ${cards || '<p class="board-empty">Chưa có ứng viên nào</p>'}
        </div>
      </section>`;
  }).join("");
}

/* ==========================================================================
   RENDER RESULTS & METRICS
   ========================================================================== */
function renderResults(results) {
  if (!results || results.length === 0) return;

  loadDecisions();

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

/** Xoá từ khoá tìm kiếm và render lại. */
function clearSearch() {
  const input = document.getElementById("searchCandidate");
  input.value = "";
  input.focus();
  filterCandidates();
}

function filterCandidates() {
  const searchInput = document.getElementById("searchCandidate");
  const searchTerm = searchInput.value.toLowerCase();

  // Nút xoá chỉ hiện khi thực sự có từ khoá
  document.getElementById("searchClear").classList.toggle("hidden", searchTerm === "");
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

  // Kiểu board có bố cục riêng -> tách hẳn nhánh render
  if (viewMode === "board") {
    renderBoard(filtered);
    return;
  }

  const container = document.getElementById("candidatesList");

  if (filtered.length === 0) {
    const msg = searchTerm
      ? `Không có ứng viên nào khớp với "${escapeHtml(searchInput.value)}"`
      : "Không tìm thấy ứng viên phù hợp với bộ lọc";
    container.innerHTML = `
      <div class="empty-filter-state">
        <i class="fa-solid fa-filter-circle-xmark"></i>
        <p>${msg}</p>
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

  // Cảnh báo CV parse thiếu
  const compSection = document.getElementById("modalCompletenessSection");
  const comp = item.completeness;
  if (comp && comp.warnings && comp.warnings.length > 0) {
    compSection.classList.remove("hidden");
    document.getElementById("modalCompleteness").innerHTML = `
      <i class="fa-solid fa-triangle-exclamation"></i>
      <div>
        <strong>Điểm số của hồ sơ này có thể không đáng tin.</strong>
        <span>${comp.warnings.map(escapeHtml).join(". ")}. Nên mở CV gốc kiểm tra lại.</span>
      </div>`;
  } else {
    compSection.classList.add("hidden");
  }

  // Điểm từng tiêu chí
  document.getElementById("modalCriteria").innerHTML =
    criterionBars(item.criterion_scores) ||
    '<p class="text-sub">Không có dữ liệu chi tiết theo tiêu chí</p>';

  // Bổ sung kỹ năng nào thì điểm tăng nhiều nhất
  const gapSection = document.getElementById("modalGapImpactSection");
  const gaps = item.skill_gaps || [];
  if (gaps.length > 0) {
    gapSection.classList.remove("hidden");
    document.getElementById("modalGapImpact").innerHTML = gaps.map(g => `
      <div class="gap-impact-row">
        <span class="gap-skill">${escapeHtml(g.skill)}</span>
        <span class="pill-tag ${g.is_required ? "orange" : "blue"}">
          ${g.is_required ? "bắt buộc" : "ưu tiên"}
        </span>
        <span class="gap-current">đang đáp ứng ${Math.round((g.current_coverage || 0) * 100)}%</span>
        <span class="gap-gain">+${g.score_gain} điểm</span>
      </div>`).join("");
  } else {
    gapSection.classList.add("hidden");
  }

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
