const API = '/api/v1';
const TOKEN_KEY = 'auth_token';
const LOGIN_KEY = 'auth_login';
const LEGACY_EMAIL_KEY = 'auth_email';

// 28 эмоциональных меток из GoEmotions
const EMOTION_COLORS = {
  admiration: '#f4a261', amusement: '#ffd93d', anger: '#d9534f',
  annoyance: '#e76f51', approval: '#2a9d8f', caring: '#e9c46a',
  confusion: '#8d99ae', curiosity: '#83c5be', desire: '#bc6c25',
  disappointment: '#7d8597', disapproval: '#a8554f', disgust: '#5d8b5a',
  embarrassment: '#c08497', excitement: '#f39c12', fear: '#6f42c1',
  gratitude: '#06d6a0', grief: '#264653', joy: '#ffd93d',
  love: '#ef476f', nervousness: '#9c89b8', optimism: '#80ed99',
  pride: '#ffb703', realization: '#48cae4', relief: '#90be6d',
  remorse: '#577590', sadness: '#4a90d9', surprise: '#f3722c',
  trust: '#17a2b8', neutral: '#adb5bd',
};

const EMOTION_RU = {
  admiration: 'Восхищение', amusement: 'Веселье', anger: 'Гнев',
  annoyance: 'Раздражение', approval: 'Одобрение', caring: 'Забота',
  confusion: 'Замешательство', curiosity: 'Любопытство', desire: 'Желание',
  disappointment: 'Разочарование', disapproval: 'Неодобрение', disgust: 'Отвращение',
  embarrassment: 'Смущение', excitement: 'Возбуждение', fear: 'Страх',
  gratitude: 'Благодарность', grief: 'Горе', joy: 'Радость',
  love: 'Любовь', nervousness: 'Нервозность', optimism: 'Оптимизм',
  pride: 'Гордость', realization: 'Осознание', relief: 'Облегчение',
  remorse: 'Раскаяние', sadness: 'Грусть', surprise: 'Удивление',
  trust: 'Доверие', neutral: 'Нейтрально',
};

let timelineChart = null;
let bookEmotionsChart = null;
let blockTimelineChart = null;
let refreshInterval = null;
let llmAvailable = false;

function getToken() { return localStorage.getItem(TOKEN_KEY); }
function setAuth(token, login) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(LOGIN_KEY, login);
  localStorage.removeItem(LEGACY_EMAIL_KEY);
}
function clearAuth() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(LOGIN_KEY);
  localStorage.removeItem(LEGACY_EMAIL_KEY);
}

async function api(path, options = {}) {
  const token = getToken();
  const headers = options.headers || {};
  if (token) headers['Authorization'] = 'Bearer ' + token;
  if (options.body && !(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }
  const res = await fetch(API + path, { ...options, headers });
  if (res.status === 401 && token) {
    clearAuth();
    renderView();
    throw new Error('Сессия истекла, войдите заново');
  }
  if (!res.ok) {
    let msg = `Ошибка ${res.status}`;
    try { const data = await res.json(); msg = data.error || msg; } catch { }
    throw new Error(msg);
  }
  if (res.status === 204) return null;
  return res.json();
}

function showToast(message, type = 'info') {
  const toast = document.getElementById('toast');
  toast.textContent = message;
  toast.className = 'toast ' + type;
  setTimeout(() => toast.classList.add('hidden'), 3000);
}

function formatDate(isoString) {
  try { return new Date(isoString).toLocaleString('ru-RU'); }
  catch { return isoString; }
}

function emotionLabel(key) { return EMOTION_RU[key] || key; }
function emotionColor(key) { return EMOTION_COLORS[key] || '#888'; }

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = String(str ?? '');
  return div.innerHTML;
}

function isolateDatasetOnLegendClick(_evt, legendItem, legend) {
  const chart = legend.chart;
  const clicked = legendItem.datasetIndex;
  const visible = chart.data.datasets.map((_, i) => !chart.getDatasetMeta(i).hidden);
  const visibleCount = visible.filter(Boolean).length;
  const isOnlyVisible = visibleCount === 1 && visible[clicked];

  chart.data.datasets.forEach((_, i) => {
    const meta = chart.getDatasetMeta(i);
    meta.hidden = isOnlyVisible ? false : i !== clicked;
  });
  chart.update();
}

function renderView() {
  const authView = document.getElementById('auth-view');
  const mainView = document.getElementById('main-view');
  const userBar = document.getElementById('user-bar');
  const token = getToken();

  if (token) {
    authView.classList.add('hidden');
    mainView.classList.remove('hidden');
    userBar.classList.remove('hidden');
    document.getElementById('user-login').textContent = localStorage.getItem(LOGIN_KEY) || '';
    loadLLMStatus();
    loadJobs();
    startAutoRefresh();
  } else {
    authView.classList.remove('hidden');
    mainView.classList.add('hidden');
    userBar.classList.add('hidden');
    updateLLMOption({ available: false, checking: false });
    stopAutoRefresh();
  }
}

function updateLLMOption(status) {
  const form = document.getElementById('upload-form');
  const input = form?.use_llm;
  const label = document.getElementById('use-llm-label');
  if (!input || !label) return;

  const checking = Boolean(status.checking);
  llmAvailable = Boolean(status.available);
  input.disabled = checking || !llmAvailable;
  if (!llmAvailable) input.checked = false;

  label.classList.toggle('disabled', input.disabled);
  label.setAttribute('aria-disabled', input.disabled ? 'true' : 'false');
  if (checking) {
    label.title = 'Проверяем доступность дополнительной модели на сервере';
  } else if (!llmAvailable) {
    label.title = 'LLM сейчас недоступна';
  } else {
    label.title = '';
  }
}

async function loadLLMStatus() {
  updateLLMOption({ available: false, checking: true });
  try {
    const status = await api('/llm/status');
    updateLLMOption(status);
  } catch {
    updateLLMOption({ available: false, checking: false });
  }
}

document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    const target = tab.dataset.tab;
    document.getElementById('login-form').classList.toggle('hidden', target !== 'login');
    document.getElementById('register-form').classList.toggle('hidden', target !== 'register');
  });
});

document.getElementById('login-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  const errorEl = form.querySelector('.form-error');
  errorEl.textContent = '';
  try {
    const data = await api('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ login: form.login.value, password: form.password.value }),
    });
    setAuth(data.token, form.login.value);
    renderView();
    showToast('Добро пожаловать!', 'success');
  } catch (err) { errorEl.textContent = err.message; }
});

document.getElementById('register-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  const errorEl = form.querySelector('.form-error');
  errorEl.textContent = '';
  try {
    await api('/auth/register', {
      method: 'POST',
      body: JSON.stringify({
        login: form.login.value,
        password: form.password.value,
      }),
    });
    const login = await api('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ login: form.login.value, password: form.password.value }),
    });
    setAuth(login.token, form.login.value);
    renderView();
    showToast('Регистрация прошла успешно!', 'success');
  } catch (err) { errorEl.textContent = err.message; }
});

document.getElementById('logout-btn').addEventListener('click', () => {
  clearAuth();
  renderView();
});

document.getElementById('upload-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  const errorEl = form.querySelector('.form-error');
  errorEl.textContent = '';
  const submitBtn = form.querySelector('button[type="submit"]');
  submitBtn.disabled = true;

  const formData = new FormData();
  formData.append('title', form.title.value);
  formData.append('file', form.file.files[0]);
  formData.append('use_llm', llmAvailable && form.use_llm.checked ? 'true' : 'false');
  formData.append('use_paragraph_analysis', form.use_paragraph_analysis.checked ? 'true' : 'false');

  try {
    await api('/jobs', { method: 'POST', body: formData });
    form.reset();
    showToast('Задача создана и поставлена в очередь', 'success');
    loadJobs();
  } catch (err) {
    errorEl.textContent = err.message;
  } finally {
    submitBtn.disabled = false;
  }
});

async function loadJobs() {
  const container = document.getElementById('jobs-list');
  try {
    const data = await api('/jobs');
    const jobs = data.jobs || [];
    if (jobs.length === 0) {
      container.innerHTML = '<div class="empty-state">Задач пока нет. Загрузите текст для анализа.</div>';
      return;
    }
    jobs.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    container.innerHTML = jobs.map(job => `
      <div class="job-item ${job.status}" data-job-id="${job.id}">
        <div class="job-status">${statusLabel(job.status)}</div>
        <div class="job-title">${escapeHtml(job.title || '(без названия)')}</div>
        <div class="job-id">${job.id.slice(0, 8)} · ${formatDate(job.created_at)}</div>
        ${job.error ? `<div class="form-error">${escapeHtml(job.error)}</div>` : ''}
      </div>
    `).join('');
    container.querySelectorAll('.job-item').forEach(el => {
      el.addEventListener('click', () => {
        const jobId = el.dataset.jobId;
        const job = jobs.find(j => j.id === jobId);
        if (job.status === 'completed') {
          loadResult(jobId, job.title);
        } else if (job.status === 'failed') {
          showToast('Задача завершилась ошибкой: ' + (job.error || 'неизвестно'), 'error');
        } else {
          showToast('Задача еще обрабатывается', 'info');
        }
      });
    });
  } catch (err) {
    if (err.message.includes('401')) { clearAuth(); renderView(); return; }
    container.innerHTML = `<div class="form-error">${escapeHtml(err.message)}</div>`;
  }
}

function statusLabel(status) {
  return {
    pending: 'В очереди',
    processing: 'Обрабатывается',
    completed: 'Готово',
    failed: 'Ошибка',
  }[status] || status;
}

document.getElementById('refresh-jobs').addEventListener('click', loadJobs);

function startAutoRefresh() {
  if (refreshInterval) return;
  refreshInterval = setInterval(loadJobs, 3000);
}

function stopAutoRefresh() {
  if (refreshInterval) { clearInterval(refreshInterval); refreshInterval = null; }
}

async function loadResult(jobId, title) {
  try {
    const wrapper = await api(`/jobs/${jobId}/result`);
    const result = wrapper.result;
    if (!result) {
      showToast('Результат имеет неожиданный формат', 'error');
      return;
    }
    if (!result.chapters || result.chapters.length === 0) {
      showToast('Не удалось проанализировать текст: главы не определены', 'error');
      return;
    }
    renderResult(result, title);
  } catch (err) {
    showToast(err.message, 'error');
  }
}

function renderResult(result, title) {
  const view = document.getElementById('result-view');
  view.classList.remove('hidden');
  view.scrollIntoView({ behavior: 'smooth' });

  // Бэк возвращает главы в порядке документа, но явно закрепляем порядок, чтобы графики и список всегда совпадали
  const chapters = (result.chapters || [])
    .slice()
    .sort((a, b) => (a.chapter_index ?? 0) - (b.chapter_index ?? 0));
  const numChapters = result.num_chapters ?? chapters.length;
  const splitMethod = result.split_method ? ` · ${result.split_method}` : '';
  const titleText = title ? `${title} · ` : '';
  const titleEl = document.getElementById('result-title');
  titleEl.textContent = `${titleText}${numChapters} глав${splitMethod}`;
  if (result.translated) {
    const badge = document.createElement('span');
    badge.className = 'translation-badge';
    badge.title = 'Текст был автоматически переведён с русского для анализа';
    badge.textContent = 'переведено с русского';
    titleEl.appendChild(badge);
  }
  if (result.use_llm) {
    const badge = document.createElement('span');
    badge.className = 'translation-badge';
    badge.title = 'При скоринге использовалась дополнительная LLM для анализа сцен';
    badge.textContent = 'LLM-анализ сцен';
    titleEl.appendChild(badge);
  }
  if (result.use_paragraph_analysis) {
    const badge = document.createElement('span');
    badge.className = 'translation-badge';
    badge.title = 'Для этого произведения сохранена эмоциональная динамика по логическим блокам';
    badge.textContent = 'анализ по логическим блокам';
    titleEl.appendChild(badge);
  }

  renderBookEmotions(result.book_level_top_emotions || []);
  renderTimelineChart(chapters);
  renderChaptersList(chapters);
  renderBlocks(result.use_paragraph_analysis ? (result.blocks || []) : []);
}

function renderBookEmotions(topEmotions) {
  const ctx = document.getElementById('book-emotions-chart').getContext('2d');
  if (bookEmotionsChart) bookEmotionsChart.destroy();

  const items = topEmotions.slice(0, 10);
  bookEmotionsChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: items.map(e => emotionLabel(e.label)),
      datasets: [{
        data: items.map(e => e.score),
        backgroundColor: items.map(e => emotionColor(e.label)),
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'Средний балл' }, beginAtZero: true },
      },
    },
  });
}

function renderTimelineChart(chapters) {
  const ctx = document.getElementById('timeline-chart').getContext('2d');
  if (timelineChart) timelineChart.destroy();

  const relevant = new Set();
  chapters.forEach(ch => (ch.top_emotions || []).slice(0, 3).forEach(e => relevant.add(e.label)));
  const emotions = [...relevant];

  const labels = chapters.map(ch => String(ch.chapter_index));
  const datasets = emotions.map(emotion => ({
    label: emotionLabel(emotion),
    data: chapters.map(ch => (ch.scores && ch.scores[emotion]) ?? 0),
    borderColor: emotionColor(emotion),
    backgroundColor: emotionColor(emotion) + '40',
    tension: 0.3,
    pointRadius: 3,
  }));

  timelineChart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          onClick: isolateDatasetOnLegendClick,
        },
      },
      scales: {
        y: { title: { display: true, text: 'Балл эмоции' }, beginAtZero: true },
      },
      interaction: { mode: 'index', intersect: false },
    },
  });
}

function renderChaptersList(chapters) {
  const container = document.getElementById('chapters-list');
  if (!chapters.length) {
    container.innerHTML = '<div class="empty-state">Нет глав</div>';
    return;
  }
  container.innerHTML = chapters.map(ch => {
    const top = (ch.top_emotions || []).slice(0, 5);
    const tags = top.map(e => `
      <span class="emotion-tag" style="background: ${emotionColor(e.label)}">
        ${escapeHtml(emotionLabel(e.label))} · ${(e.score * 100).toFixed(1)}%
      </span>
    `).join('');
    const dominant = top[0];
    const borderColor = dominant ? emotionColor(dominant.label) : '#888';
    // Номера страниц имеют смысл только для PDF, для TXT модель возвращает 1–1, поэтому скрываем такой диапазон
    const hasRealPages = ch.start_page && ch.end_page && !(ch.start_page === 1 && ch.end_page === 1);
    const pages = hasRealPages ? ` · стр. ${ch.start_page}–${ch.end_page}` : '';
    return `
      <div class="chapter-item" style="border-left-color: ${borderColor}">
        <div class="chapter-header">
          <span class="chapter-index">${ch.chapter_index}:</span>
          <span class="chapter-title">${escapeHtml(ch.chapter_title || '')}</span>
          <span class="chapter-meta">${ch.chapter_char_count || 0} симв.${pages}</span>
        </div>
        <div class="emotion-tags">${tags}</div>
      </div>
    `;
  }).join('');
}

function renderBlocks(blocks) {
  const section = document.getElementById('blocks-section');
  if (!blocks.length) {
    section.classList.add('hidden');
    if (blockTimelineChart) {
      blockTimelineChart.destroy();
      blockTimelineChart = null;
    }
    return;
  }

  section.classList.remove('hidden');
  renderBlockTimelineChart(blocks);
  renderBlocksList(blocks);
}

function renderBlockTimelineChart(blocks) {
  const ctx = document.getElementById('block-timeline-chart').getContext('2d');
  if (blockTimelineChart) blockTimelineChart.destroy();

  const relevant = new Set();
  blocks.forEach(block => (block.top_emotions || []).slice(0, 3).forEach(e => relevant.add(e.label)));
  const emotions = [...relevant];
  const labels = blocks.map(block => `Блок ${block.block_index}`);
  const datasets = emotions.map(emotion => ({
    label: emotionLabel(emotion),
    data: blocks.map(block => (block.scores && block.scores[emotion]) ?? 0),
    borderColor: emotionColor(emotion),
    backgroundColor: emotionColor(emotion) + '40',
    tension: 0.25,
    pointRadius: blocks.length > 80 ? 0 : 2,
  }));

  blockTimelineChart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          onClick: isolateDatasetOnLegendClick,
        },
      },
      scales: {
        y: { title: { display: true, text: 'Балл эмоции' }, beginAtZero: true },
        x: { ticks: { maxTicksLimit: 16 } },
      },
      interaction: { mode: 'index', intersect: false },
    },
  });
}

function renderBlocksList(blocks) {
  const container = document.getElementById('blocks-list');
  container.innerHTML = blocks.map(block => {
    const top = (block.top_emotions || []).slice(0, 4);
    const tags = top.map(e => `
      <span class="emotion-tag" style="background: ${emotionColor(e.label)}">
        ${escapeHtml(emotionLabel(e.label))} · ${(e.score * 100).toFixed(1)}%
      </span>
    `).join('');
    const dominant = top[0];
    const borderColor = dominant ? emotionColor(dominant.label) : '#888';
    return `
      <div class="block-item" style="border-left-color: ${borderColor}">
        <div class="chapter-header">
          <span class="chapter-index">Блок ${block.block_index}</span>
          <span class="chapter-title">Глава ${block.chapter_index}, сцена ${block.scene_index}</span>
          <span class="chapter-meta">${block.scene_word_count || 0} слов</span>
        </div>
        <div class="emotion-tags">${tags}</div>
      </div>
    `;
  }).join('');
}

document.getElementById('close-result').addEventListener('click', () => {
  document.getElementById('result-view').classList.add('hidden');
});

renderView();
