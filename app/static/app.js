/**
 * NIFTY 50 Stock Rating System — Multi-Page SPA v2
 * Pages: Dashboard | Stock Analysis | Indicators | Model Performance
 */

const API = window.location.origin;
let allStocks = [];
let allIndicators = [];
let currentPage = 'dashboard';
let sortCol = 'composite_rating';
let sortAsc = false;
let currentFilter = 'all';
let activeCharts = {};

// ============================================================
// Init
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    setupNav();
    setupModal();
    setupSearch();
    navigateTo('dashboard');
});

// ============================================================
// Navigation
// ============================================================
function setupNav() {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => navigateTo(item.dataset.page));
    });
    document.getElementById('menu-toggle').addEventListener('click', () => {
        document.getElementById('sidebar').classList.toggle('open');
    });
}

function navigateTo(page) {
    currentPage = page;
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    const navEl = document.querySelector(`[data-page="${page}"]`);
    if (navEl) navEl.classList.add('active');

    const titles = { dashboard:'Dashboard', stock:'Stock Analysis', indicators:'Technical Indicators', model:'Model Performance' };
    document.getElementById('page-title').textContent = titles[page] || page;

    destroyCharts();
    const container = document.getElementById('page-container');
    container.innerHTML = '<div class="loading-container"><div class="spinner"></div><div class="loading-text">Loading...</div></div>';

    switch(page) {
        case 'dashboard': loadDashboard(); break;
        case 'stock': loadStockPage(); break;
        case 'indicators': loadIndicatorsPage(); break;
        case 'model': loadModelPage(); break;
    }
}

function destroyCharts() {
    Object.values(activeCharts).forEach(c => { try { if(c.remove) c.remove(); else if(c.destroy) c.destroy(); } catch(e){} });
    activeCharts = {};
}

// ============================================================
// Search
// ============================================================
function setupSearch() {
    let t;
    document.getElementById('global-search').addEventListener('input', () => {
        clearTimeout(t);
        t = setTimeout(() => {
            if (currentPage === 'dashboard' || currentPage === 'indicators') renderCurrentPage();
        }, 200);
    });
}

function renderCurrentPage() {
    if (currentPage === 'dashboard') renderDashboardTable();
    if (currentPage === 'indicators') renderIndicatorsTable();
}

function getSearchQuery() { return document.getElementById('global-search').value.toLowerCase(); }

// ============================================================
// Modal
// ============================================================
function setupModal() {
    document.getElementById('modal-close').addEventListener('click', closeModal);
    document.getElementById('stock-modal').addEventListener('click', e => {
        if (e.target.id === 'stock-modal') closeModal();
    });
    document.addEventListener('keydown', e => { if(e.key==='Escape') closeModal(); });
}

function closeModal() {
    document.getElementById('stock-modal').classList.remove('active');
    destroyCharts();
}

// ============================================================
// PAGE: Dashboard
// ============================================================
async function loadDashboard() {
    try {
        const res = await fetch(`${API}/api/ratings`);
        const data = await res.json();
        allStocks = data.stocks || [];
        document.getElementById('sidebar-status-text').textContent = `${allStocks.length} stocks`;
        renderDashboardPage();
    } catch(err) {
        document.getElementById('page-container').innerHTML = `
            <div class="empty-state"><div class="empty-icon">⚠️</div>
            <div class="empty-title">API Offline</div>
            <div class="empty-desc">Could not connect. Make sure uvicorn is running.</div></div>`;
    }
}

function renderDashboardPage() {
    const c = document.getElementById('page-container');

    // Stats
    const ratings = allStocks.map(s=>s.composite_rating).filter(r=>r>0);
    const avg = ratings.length ? (ratings.reduce((a,b)=>a+b,0)/ratings.length).toFixed(1) : '—';
    const top = allStocks.filter(s=>s.composite_rating>=8).length;
    const bottom = allStocks.filter(s=>s.composite_rating<=3).length;
    const bullish = allStocks.filter(s=>(s.pred_30||0)>0).length;

    c.innerHTML = `
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-label">Total Stocks</div><div class="stat-value">${allStocks.length}</div></div>
            <div class="stat-card"><div class="stat-label">Avg Rating</div><div class="stat-value">${avg}</div></div>
            <div class="stat-card"><div class="stat-label">Top Rated</div><div class="stat-value">${top}</div><div class="stat-sub up">Rating ≥ 8</div></div>
            <div class="stat-card"><div class="stat-label">Bullish (30D)</div><div class="stat-value">${bullish}</div><div class="stat-sub up">Positive outlook</div></div>
        </div>
        <div class="filter-bar" id="dash-filters">
            <span class="filter-chip active" data-f="all">All Stocks</span>
            <span class="filter-chip" data-f="top">Top (8-10)</span>
            <span class="filter-chip" data-f="mid">Mid (5-7)</span>
            <span class="filter-chip" data-f="low">Avoid (1-4)</span>
            <span class="filter-chip" data-f="bullish">Most Bullish</span>
            <span class="filter-chip" data-f="bearish">Most Bearish</span>
        </div>
        <div class="card"><div class="card-header"><div class="card-title">📊 Stock Rankings</div></div>
            <div class="card-body-flush" id="dash-table-body"></div>
        </div>`;

    document.querySelectorAll('#dash-filters .filter-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            document.querySelectorAll('#dash-filters .filter-chip').forEach(c=>c.classList.remove('active'));
            chip.classList.add('active');
            currentFilter = chip.dataset.f;
            renderDashboardTable();
        });
    });

    renderDashboardTable();
}

function renderDashboardTable() {
    const q = getSearchQuery();
    let stocks = [...allStocks];
    if(q) stocks = stocks.filter(s => s.name.toLowerCase().includes(q) || s.ticker.toLowerCase().includes(q));

    switch(currentFilter) {
        case 'top': stocks = stocks.filter(s=>s.composite_rating>=8); break;
        case 'mid': stocks = stocks.filter(s=>s.composite_rating>=5&&s.composite_rating<=7); break;
        case 'low': stocks = stocks.filter(s=>s.composite_rating<=4); break;
        case 'bullish': stocks.sort((a,b)=>(b.pred_30||0)-(a.pred_30||0)); stocks=stocks.slice(0,15); break;
        case 'bearish': stocks.sort((a,b)=>(a.pred_30||0)-(b.pred_30||0)); stocks=stocks.slice(0,15); break;
    }

    stocks.sort((a,b) => {
        let va=a[sortCol]??0, vb=b[sortCol]??0;
        if(sortCol==='name') return sortAsc ? a.name.localeCompare(b.name) : b.name.localeCompare(a.name);
        return sortAsc ? va-vb : vb-va;
    });

    if(!stocks.length) {
        document.getElementById('dash-table-body').innerHTML = '<div class="empty-state"><div class="empty-icon">📭</div><div class="empty-title">No matches</div></div>';
        return;
    }

    let html = `<table class="data-table"><thead><tr>
        <th data-s="rank">#</th><th data-s="name">Stock</th><th data-s="composite_rating">Rating</th>
        <th data-s="pred_10">10D Return</th><th data-s="pred_20">20D Return</th><th data-s="pred_30">30D Return</th>
    </tr></thead><tbody>`;

    stocks.forEach((s,i) => {
        const d = Math.min(i*25,500);
        html += `<tr class="fade-in" style="animation-delay:${d}ms" onclick="openStockQuickView('${s.name}')">
            <td style="color:var(--text-muted);font-weight:600">${i+1}</td>
            <td><div class="ticker-cell"><div class="ticker-avatar">${s.name.substring(0,2)}</div><div><div class="ticker-name">${s.name}</div><div class="ticker-symbol">${s.ticker}</div></div></div></td>
            <td><span class="rating-badge rating-${s.composite_rating}">${s.composite_rating}</span></td>
            <td><span class="${(s.pred_10||0)>=0?'val-positive':'val-negative'}">${fmtPct(s.pred_10)}</span></td>
            <td><span class="${(s.pred_20||0)>=0?'val-positive':'val-negative'}">${fmtPct(s.pred_20)}</span></td>
            <td><span class="${(s.pred_30||0)>=0?'val-positive':'val-negative'}">${fmtPct(s.pred_30)}</span></td>
        </tr>`;
    });
    html += '</tbody></table>';
    document.getElementById('dash-table-body').innerHTML = html;

    document.querySelectorAll('.data-table thead th').forEach(th => {
        th.addEventListener('click', () => {
            const col = th.dataset.s;
            if(col===sortCol) sortAsc=!sortAsc; else { sortCol=col; sortAsc=false; }
            renderDashboardTable();
        });
    });
}

// ============================================================
// PAGE: Stock Analysis (Candlestick Charts)
// ============================================================
async function loadStockPage() {
    const c = document.getElementById('page-container');
    c.innerHTML = `
        <div style="margin-bottom:20px">
            <select id="stock-select" style="width:100%;padding:12px 18px;background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-md);color:var(--text-primary);font-size:0.9rem;font-family:'Inter',sans-serif;outline:none;">
                <option value="">Select a stock to analyze...</option>
            </select>
        </div>
        <div id="stock-analysis-content">
            <div class="empty-state"><div class="empty-icon">📈</div><div class="empty-title">Select a Stock</div>
            <div class="empty-desc">Choose a stock from the dropdown to view candlestick charts, predictions, and detailed analysis.</div></div>
        </div>`;

    // Populate dropdown
    try {
        const res = await fetch(`${API}/api/tickers`);
        const data = await res.json();
        const sel = document.getElementById('stock-select');
        data.tickers.forEach(t => {
            const opt = document.createElement('option');
            opt.value = t.name;
            opt.textContent = `${t.name} (${t.symbol})`;
            sel.appendChild(opt);
        });
        sel.addEventListener('change', () => { if(sel.value) loadStockAnalysis(sel.value); });
    } catch(e) {}
}

async function loadStockAnalysis(ticker) {
    const content = document.getElementById('stock-analysis-content');
    content.innerHTML = '<div class="loading-container"><div class="spinner"></div><div class="loading-text">Loading '+ticker+'...</div></div>';
    destroyCharts();

    try {
        const [detailRes, candleRes, indRes] = await Promise.all([
            fetch(`${API}/api/stock/${ticker}`),
            fetch(`${API}/api/stock/${ticker}/candles?days=730`),
            fetch(`${API}/api/stock/${ticker}/indicators`),
        ]);
        const detail = await detailRes.json();
        const candles = await candleRes.json();
        const indicators = await indRes.json();
        renderStockAnalysis(detail, candles, indicators);
    } catch(e) {
        content.innerHTML = '<div class="empty-state"><div class="empty-icon">⚠️</div><div class="empty-title">Error</div><div class="empty-desc">Failed to load data for '+ticker+'</div></div>';
    }
}

function renderStockAnalysis(detail, candles, indicators) {
    const s = detail.stats || {};
    const r = detail.rating || {};
    const chg = s.day_change || 0;
    const chgPct = s.day_change_pct || 0;
    const chgClass = chg >= 0 ? 'val-positive' : 'val-negative';

    // Indicator summary
    const ind = indicators.summary || {};
    const overallClass = getOverallClass(ind.overall);

    let html = `
        <div class="detail-grid" style="grid-template-columns:repeat(6,1fr)">
            <div class="detail-item"><div class="detail-label">Price</div><div class="detail-value">₹${s.current_price||'—'}</div></div>
            <div class="detail-item"><div class="detail-label">Change</div><div class="detail-value ${chgClass}">${chg>=0?'+':''}${chg} (${chgPct>=0?'+':''}${chgPct}%)</div></div>
            <div class="detail-item"><div class="detail-label">52W High</div><div class="detail-value val-positive">₹${s['52w_high']||'—'}</div></div>
            <div class="detail-item"><div class="detail-label">52W Low</div><div class="detail-value val-negative">₹${s['52w_low']||'—'}</div></div>
            <div class="detail-item"><div class="detail-label">Rating</div><div class="detail-value"><span class="rating-badge rating-${r.composite_rating||0}" style="font-size:1.3rem;width:48px;height:48px">${r.composite_rating||'—'}</span></div></div>
            <div class="detail-item"><div class="detail-label">Signal</div><div class="detail-value"><span class="${overallClass}">${ind.overall||'N/A'}</span></div></div>
        </div>

        <div class="grid-2" style="margin-bottom:20px">
            <div class="chart-box">
                <div class="chart-box-header">
                    <div class="chart-box-title">📈 Candlestick Chart</div>
                    <div class="tf-buttons">
                        <button class="tf-btn" onclick="setCandleDays(90)">3M</button>
                        <button class="tf-btn" onclick="setCandleDays(180)">6M</button>
                        <button class="tf-btn active" onclick="setCandleDays(365)">1Y</button>
                        <button class="tf-btn" onclick="setCandleDays(730)">2Y</button>
                    </div>
                </div>
                <div class="chart-box-body" id="candle-chart" style="height:400px"></div>
            </div>
            <div class="chart-box">
                <div class="chart-box-header"><div class="chart-box-title">📊 Volume</div></div>
                <div class="chart-box-body" id="volume-chart" style="height:400px"></div>
            </div>
        </div>

        <!-- Indicator signals -->
        <div class="card" style="margin-bottom:20px">
            <div class="card-header">
                <div class="card-title">🔬 Technical Signals — ${ind.buy||0} Buy / ${ind.sell||0} Sell / ${ind.hold||0} Hold</div>
                <div><span class="${overallClass}">${ind.overall||'N/A'}</span></div>
            </div>
            <div class="card-body-flush" id="signal-table"></div>
        </div>

        <!-- Predictions -->
        <div class="card">
            <div class="card-header"><div class="card-title">🎯 Predicted Returns</div></div>
            <div class="card-body">
                <div class="detail-grid" style="grid-template-columns:repeat(3,1fr)">
                    <div class="detail-item"><div class="detail-label">10-Day Predicted</div><div class="detail-value ${(r.pred_10||0)>=0?'val-positive':'val-negative'}">${fmtPct(r.pred_10)}</div></div>
                    <div class="detail-item"><div class="detail-label">20-Day Predicted</div><div class="detail-value ${(r.pred_20||0)>=0?'val-positive':'val-negative'}">${fmtPct(r.pred_20)}</div></div>
                    <div class="detail-item"><div class="detail-label">30-Day Predicted</div><div class="detail-value ${(r.pred_30||0)>=0?'val-positive':'val-negative'}">${fmtPct(r.pred_30)}</div></div>
                </div>
            </div>
        </div>`;

    document.getElementById('stock-analysis-content').innerHTML = html;

    // Render candlestick
    window._candleData = candles.candles;
    window._volumeData = candles.volumes;
    renderCandleChart(candles.candles, candles.volumes, 365);

    // Render signals table
    renderSignalTable(indicators.signals);
}

function renderCandleChart(allCandles, allVolumes, days) {
    destroyCharts();
    const candles = allCandles.slice(-days);
    const volumes = allVolumes.slice(-days);

    // Candle chart
    const candleEl = document.getElementById('candle-chart');
    if (!candleEl) return;
    candleEl.innerHTML = '';
    const chart = LightweightCharts.createChart(candleEl, {
        width: candleEl.clientWidth,
        height: 388,
        layout: { background: { color: '#1a1f2e' }, textColor: '#94a3b8' },
        grid: { vertLines: { color: 'rgba(255,255,255,0.03)' }, horzLines: { color: 'rgba(255,255,255,0.03)' } },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: { borderColor: 'rgba(255,255,255,0.06)' },
        timeScale: { borderColor: 'rgba(255,255,255,0.06)', timeVisible: false },
    });
    const candleSeries = chart.addCandlestickSeries({
        upColor: '#10b981', downColor: '#ef4444',
        borderUpColor: '#10b981', borderDownColor: '#ef4444',
        wickUpColor: '#10b981', wickDownColor: '#ef4444',
    });
    candleSeries.setData(candles);
    chart.timeScale().fitContent();
    activeCharts.candle = chart;

    // Volume chart
    const volEl = document.getElementById('volume-chart');
    if (!volEl) return;
    volEl.innerHTML = '';
    const vChart = LightweightCharts.createChart(volEl, {
        width: volEl.clientWidth,
        height: 388,
        layout: { background: { color: '#1a1f2e' }, textColor: '#94a3b8' },
        grid: { vertLines: { color: 'rgba(255,255,255,0.03)' }, horzLines: { color: 'rgba(255,255,255,0.03)' } },
        rightPriceScale: { borderColor: 'rgba(255,255,255,0.06)' },
        timeScale: { borderColor: 'rgba(255,255,255,0.06)' },
    });
    const volSeries = vChart.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: '' });
    volSeries.setData(volumes);
    vChart.timeScale().fitContent();
    activeCharts.volume = vChart;

    // Resize handler
    const ro = new ResizeObserver(() => {
        chart.applyOptions({ width: candleEl.clientWidth });
        vChart.applyOptions({ width: volEl.clientWidth });
    });
    ro.observe(candleEl);
    ro.observe(volEl);
}

window.setCandleDays = function(days) {
    document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    if (window._candleData) renderCandleChart(window._candleData, window._volumeData, days);
};

function renderSignalTable(signals) {
    if (!signals || !Object.keys(signals).length) return;
    let html = '<table class="data-table"><thead><tr><th>Indicator</th><th>Value</th><th>Signal</th><th>Action</th></tr></thead><tbody>';
    Object.entries(signals).forEach(([name, s]) => {
        const actionClass = s.action === 'buy' ? 'signal-buy' : (s.action === 'sell' ? 'signal-sell' : 'signal-hold');
        html += `<tr><td style="font-weight:600">${name.replace(/_/g,' ')}</td>
            <td class="val-neutral">${s.value}</td>
            <td style="color:var(--text-secondary);font-size:0.82rem">${s.signal}</td>
            <td><span class="${actionClass}">${s.action.toUpperCase()}</span></td></tr>`;
    });
    html += '</tbody></table>';
    document.getElementById('signal-table').innerHTML = html;
}

// ============================================================
// PAGE: Indicators (All Stocks Summary)
// ============================================================
async function loadIndicatorsPage() {
    const c = document.getElementById('page-container');
    c.innerHTML = '<div class="loading-container"><div class="spinner"></div><div class="loading-text">Loading indicator data...</div></div>';

    try {
        const res = await fetch(`${API}/api/indicators/all`);
        const data = await res.json();
        allIndicators = data.stocks || [];
        renderIndicatorsPage();
    } catch(e) {
        c.innerHTML = '<div class="empty-state"><div class="empty-icon">⚠️</div><div class="empty-title">Error</div></div>';
    }
}

function renderIndicatorsPage() {
    const c = document.getElementById('page-container');

    // Stats
    const buyCount = allIndicators.filter(s => s.overall.includes('Buy')).length;
    const sellCount = allIndicators.filter(s => s.overall.includes('Sell')).length;
    const holdCount = allIndicators.filter(s => s.overall === 'Hold').length;

    c.innerHTML = `
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-label">Stocks Analyzed</div><div class="stat-value">${allIndicators.length}</div></div>
            <div class="stat-card"><div class="stat-label">Buy Signals</div><div class="stat-value" style="color:var(--green);-webkit-text-fill-color:var(--green)">${buyCount}</div></div>
            <div class="stat-card"><div class="stat-label">Sell Signals</div><div class="stat-value" style="color:var(--red);-webkit-text-fill-color:var(--red)">${sellCount}</div></div>
            <div class="stat-card"><div class="stat-label">Hold/Neutral</div><div class="stat-value" style="color:var(--amber);-webkit-text-fill-color:var(--amber)">${holdCount}</div></div>
        </div>
        <div class="card"><div class="card-header"><div class="card-title">🔬 Technical Indicator Summary — All Stocks</div></div>
            <div class="card-body-flush" id="ind-table-body"></div>
        </div>`;

    renderIndicatorsTable();
}

function renderIndicatorsTable() {
    const q = getSearchQuery();
    let stocks = [...allIndicators];
    if (q) stocks = stocks.filter(s => s.name.toLowerCase().includes(q));

    let html = `<div style="overflow-x:auto"><table class="data-table"><thead><tr>
        <th>#</th><th>Stock</th><th>Price</th><th>RSI</th><th>MACD</th><th>ADX</th>
        <th>Stoch K</th><th>CCI</th><th>Vol%</th><th>Mom(20D)</th><th>Drawdown</th>
        <th>Buy</th><th>Sell</th><th>Overall</th>
    </tr></thead><tbody>`;

    stocks.forEach((s, i) => {
        const rsiClass = (s.rsi < 30) ? 'val-positive' : (s.rsi > 70 ? 'val-negative' : 'val-neutral');
        const overallClass = getOverallClass(s.overall);

        html += `<tr class="fade-in" style="animation-delay:${Math.min(i*20,400)}ms" onclick="openStockQuickView('${s.name}')">
            <td style="color:var(--text-muted)">${i+1}</td>
            <td><div class="ticker-cell"><div class="ticker-avatar">${s.name.substring(0,2)}</div><div><div class="ticker-name">${s.name}</div></div></div></td>
            <td class="val-neutral">₹${s.close||'—'}</td>
            <td class="${rsiClass}">${s.rsi??'—'}</td>
            <td class="${(s.macd||0)>=0?'val-positive':'val-negative'}">${s.macd??'—'}</td>
            <td class="val-neutral">${s.adx??'—'}</td>
            <td class="${(s.stoch_k||50)<20?'val-positive':((s.stoch_k||50)>80?'val-negative':'val-neutral')}">${s.stoch_k??'—'}</td>
            <td class="val-neutral">${s.cci??'—'}</td>
            <td class="val-neutral">${s.volatility??'—'}%</td>
            <td class="${(s.momentum_20||0)>=0?'val-positive':'val-negative'}">${fmtVal(s.momentum_20)}%</td>
            <td class="val-negative">${fmtVal(s.drawdown)}%</td>
            <td><span class="signal-buy">${s.buy_signals}</span></td>
            <td><span class="signal-sell">${s.sell_signals}</span></td>
            <td><span class="${overallClass}">${s.overall}</span></td>
        </tr>`;
    });
    html += '</tbody></table></div>';
    document.getElementById('ind-table-body').innerHTML = html;
}

// ============================================================
// PAGE: Model Performance
// ============================================================
async function loadModelPage() {
    const c = document.getElementById('page-container');
    c.innerHTML = '<div class="loading-container"><div class="spinner"></div><div class="loading-text">Loading model metrics...</div></div>';

    try {
        const [statusRes, accRes] = await Promise.all([
            fetch(`${API}/api/model/status`),
            fetch(`${API}/api/model/accuracy`),
        ]);
        const status = await statusRes.json();
        const accuracy = await accRes.json();
        renderModelPage(status, accuracy);
    } catch(e) {
        c.innerHTML = '<div class="empty-state"><div class="empty-icon">⚠️</div><div class="empty-title">Error loading model data</div></div>';
    }
}

function renderModelPage(status, accuracy) {
    const c = document.getElementById('page-container');

    // Model status
    let statusHtml = '<div class="grid-3" style="margin-bottom:24px">';
    ['fwd_return_10','fwd_return_20','fwd_return_30'].forEach(target => {
        const lgbm = status.lgbm[target] || {};
        const tft = status.tft[target] || {};
        statusHtml += `
            <div class="card">
                <div class="card-header"><div class="card-title">${target.replace('fwd_return_','')}-Day Horizon</div></div>
                <div class="card-body">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                        <span style="font-weight:600;font-size:0.85rem">LightGBM</span>
                        <span class="${lgbm.trained?'chip-trained':'chip-not-trained'}">${lgbm.trained?'✓ Trained':'✗ Not Trained'}</span>
                    </div>
                    ${lgbm.trained ? `<div style="font-size:0.75rem;color:var(--text-muted)">Size: ${lgbm.size_mb} MB</div>` : ''}
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-top:12px">
                        <span style="font-weight:600;font-size:0.85rem">TFT (Transformer)</span>
                        <span class="${tft.trained?'chip-trained':'chip-not-trained'}">${tft.trained?'✓ Trained':'✗ Not Trained'}</span>
                    </div>
                </div>
            </div>`;
    });
    statusHtml += '</div>';

    // Metrics
    let metricsHtml = '';
    if (accuracy.metrics && accuracy.metrics.length) {
        metricsHtml = '<div class="grid-3" style="margin-bottom:24px">';
        accuracy.metrics.forEach(m => {
            const horizon = m.target.replace('fwd_return_', '');
            metricsHtml += `
                <div class="model-metric-card">
                    <div class="metric-label">${horizon}-Day Model Accuracy</div>
                    <div class="metric-big" style="color:var(--green)">${m.directional_accuracy}%</div>
                    <div class="metric-target">Directional Accuracy</div>
                    <div style="margin-top:16px;display:flex;gap:20px;justify-content:center">
                        <div><div class="metric-label">RMSE</div><div style="font-size:1.1rem;font-weight:700;color:var(--cyan);font-family:'JetBrains Mono',monospace">${m.rmse}</div></div>
                        <div><div class="metric-label">Spearman IC</div><div style="font-size:1.1rem;font-weight:700;color:var(--purple);font-family:'JetBrains Mono',monospace">${m.spearman_ic}</div></div>
                    </div>
                </div>`;
        });
        metricsHtml += '</div>';
    }

    // Parameters
    let paramsHtml = '<div class="grid-2" style="margin-bottom:24px">';

    // LightGBM params
    const lgbmCfg = accuracy.current_config?.lgbm || {};
    paramsHtml += `<div class="card"><div class="card-header"><div class="card-title">⚙️ LightGBM Parameters</div></div>
        <div class="card-body-flush"><table class="param-table"><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>`;
    Object.entries(lgbmCfg).forEach(([k,v]) => {
        paramsHtml += `<tr><td>${k}</td><td>${v}</td></tr>`;
    });
    paramsHtml += '</tbody></table></div></div>';

    // TFT params
    const tftCfg = accuracy.current_config?.tft || {};
    paramsHtml += `<div class="card"><div class="card-header"><div class="card-title">⚙️ TFT Parameters</div></div>
        <div class="card-body-flush"><table class="param-table"><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>`;
    Object.entries(tftCfg).forEach(([k,v]) => {
        paramsHtml += `<tr><td>${k}</td><td>${v}</td></tr>`;
    });
    paramsHtml += '</tbody></table></div></div>';
    paramsHtml += '</div>';

    // Best params from Optuna (if available)
    let bestParamsHtml = '';
    if (accuracy.best_params && Object.keys(accuracy.best_params).length) {
        bestParamsHtml = '<div class="card" style="margin-bottom:24px"><div class="card-header"><div class="card-title">🏆 Best Parameters (Optuna Tuned)</div></div><div class="card-body-flush"><table class="param-table"><thead><tr><th>Target</th><th>Parameter</th><th>Best Value</th></tr></thead><tbody>';
        Object.entries(accuracy.best_params).forEach(([target, params]) => {
            const horizon = target.replace('fwd_return_', '');
            Object.entries(params).forEach(([k,v],i) => {
                bestParamsHtml += `<tr><td>${i===0?horizon+'D':'&nbsp;'}</td><td>${k}</td><td>${typeof v==='number' ? v.toFixed(6) : v}</td></tr>`;
            });
        });
        bestParamsHtml += '</tbody></table></div></div>';
    }

    // Charts
    let chartsHtml = '';
    if (accuracy.charts && accuracy.charts.length) {
        chartsHtml = '<div class="card"><div class="card-header"><div class="card-title">📈 Evaluation Charts</div></div><div class="card-body"><div class="grid-3">';
        accuracy.charts.forEach(chart => {
            chartsHtml += `<div style="text-align:center"><img src="${API}/api/charts/${chart}" style="width:100%;border-radius:8px;border:1px solid var(--border)" alt="${chart}"><div style="font-size:0.75rem;color:var(--text-muted);margin-top:6px">${chart.replace('.png','').replace(/_/g,' ')}</div></div>`;
        });
        chartsHtml += '</div></div></div>';
    }

    c.innerHTML = `
        <h3 style="font-size:1rem;font-weight:700;margin-bottom:16px">🤖 Model Training Status</h3>
        ${statusHtml}
        <h3 style="font-size:1rem;font-weight:700;margin-bottom:16px">📊 Model Accuracy</h3>
        ${metricsHtml}
        ${bestParamsHtml}
        <h3 style="font-size:1rem;font-weight:700;margin-bottom:16px">⚙️ Training Configuration</h3>
        ${paramsHtml}
        ${chartsHtml}`;
}

// ============================================================
// Quick View Modal (from any page)
// ============================================================
async function openStockQuickView(ticker) {
    const modal = document.getElementById('stock-modal');
    modal.classList.add('active');
    document.getElementById('modal-ticker').textContent = ticker;
    document.getElementById('modal-subtitle').textContent = 'Loading...';
    document.getElementById('modal-body').innerHTML = '<div class="loading-container"><div class="spinner"></div></div>';

    try {
        const [detailRes, candleRes] = await Promise.all([
            fetch(`${API}/api/stock/${ticker}`),
            fetch(`${API}/api/stock/${ticker}/candles?days=365`),
        ]);
        const detail = await detailRes.json();
        const candles = await candleRes.json();
        renderQuickView(detail, candles);
    } catch(e) {
        document.getElementById('modal-body').innerHTML = '<div class="empty-state"><div class="empty-icon">⚠️</div><div class="empty-title">Error loading data</div></div>';
    }
}
window.openStockQuickView = openStockQuickView;

function renderQuickView(detail, candles) {
    const s = detail.stats || {};
    const r = detail.rating || {};
    const chg = s.day_change || 0;
    const chgClass = chg >= 0 ? 'val-positive' : 'val-negative';

    document.getElementById('modal-subtitle').textContent = `${detail.full_ticker} — ₹${s.current_price}`;

    let html = `
        <div class="detail-grid" style="grid-template-columns:repeat(5,1fr);margin-bottom:20px">
            <div class="detail-item"><div class="detail-label">Price</div><div class="detail-value">₹${s.current_price||'—'}</div></div>
            <div class="detail-item"><div class="detail-label">Change</div><div class="detail-value ${chgClass}">${chg>=0?'+':''}${s.day_change_pct}%</div></div>
            <div class="detail-item"><div class="detail-label">52W High</div><div class="detail-value val-positive">₹${s['52w_high']}</div></div>
            <div class="detail-item"><div class="detail-label">52W Low</div><div class="detail-value val-negative">₹${s['52w_low']}</div></div>
            <div class="detail-item"><div class="detail-label">Rating</div><div class="detail-value"><span class="rating-badge rating-${r.composite_rating||0}" style="width:44px;height:44px;font-size:1.2rem">${r.composite_rating||'—'}</span></div></div>
        </div>
        <div class="chart-box">
            <div class="chart-box-header"><div class="chart-box-title">📈 1-Year Candlestick</div></div>
            <div class="chart-box-body" id="modal-candle-chart" style="height:350px"></div>
        </div>
        <div class="detail-grid" style="grid-template-columns:repeat(3,1fr)">
            <div class="detail-item"><div class="detail-label">10D Predicted</div><div class="detail-value ${(r.pred_10||0)>=0?'val-positive':'val-negative'}">${fmtPct(r.pred_10)}</div></div>
            <div class="detail-item"><div class="detail-label">20D Predicted</div><div class="detail-value ${(r.pred_20||0)>=0?'val-positive':'val-negative'}">${fmtPct(r.pred_20)}</div></div>
            <div class="detail-item"><div class="detail-label">30D Predicted</div><div class="detail-value ${(r.pred_30||0)>=0?'val-positive':'val-negative'}">${fmtPct(r.pred_30)}</div></div>
        </div>`;

    document.getElementById('modal-body').innerHTML = html;

    // Render candle in modal
    setTimeout(() => {
        const el = document.getElementById('modal-candle-chart');
        if (!el) return;
        const chart = LightweightCharts.createChart(el, {
            width: el.clientWidth, height: 338,
            layout: { background:{color:'#1a1f2e'}, textColor:'#94a3b8' },
            grid: { vertLines:{color:'rgba(255,255,255,0.03)'}, horzLines:{color:'rgba(255,255,255,0.03)'} },
            rightPriceScale: { borderColor:'rgba(255,255,255,0.06)' },
            timeScale: { borderColor:'rgba(255,255,255,0.06)' },
        });
        const series = chart.addCandlestickSeries({
            upColor:'#10b981', downColor:'#ef4444',
            borderUpColor:'#10b981', borderDownColor:'#ef4444',
            wickUpColor:'#10b981', wickDownColor:'#ef4444',
        });
        series.setData(candles.candles);
        chart.timeScale().fitContent();
        activeCharts.modalCandle = chart;
    }, 100);
}

// ============================================================
// Helpers
// ============================================================
function fmtPct(v) {
    if (v === null || v === undefined) return '—';
    const pct = (v * 100).toFixed(2);
    return `${v >= 0 ? '+' : ''}${pct}%`;
}
function fmtVal(v) {
    if (v === null || v === undefined) return '—';
    return v >= 0 ? `+${v}` : `${v}`;
}
function getOverallClass(overall) {
    if (!overall) return '';
    const o = overall.toLowerCase().replace(/\s+/g, '-');
    return `overall-${o}`;
}
