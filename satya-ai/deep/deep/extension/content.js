const API_URL = "http://127.0.0.1:8000/predict";
const VERIFY_TEXT_URL = "http://127.0.0.1:8000/verify-text";
const videoData = new WeakMap();
const badges = new Map();
const scannedCache = new Map();
const FAKE_NEWS_KEYWORDS = ["breaking", "shocking", "urgent", "leak", "unbelievable", "exposed", "truth"];

window.tabStats = { scanned: 0, fake: 0, suspicious: 0, textVerified: 0 };
let extensionEnabled = true;
let currentTargetLang = "en";

// Inject Notification Banner
function injectThreatBanner() {
    if (document.getElementById('df-threat-banner')) return;
    const banner = document.createElement('div');
    banner.id = 'df-threat-banner';
    banner.className = 'df-threat-banner';
    banner.innerText = '⚠️ SECURITY ALERT: High-Confidence Synthetic Media Detected on this Page';
    document.body.appendChild(banner);
}

// Inject Modal Base for Visual Inspection
function injectInspectModal() {
    if (document.getElementById('df-inspect-modal')) return;
    const modalHTML = `
        <div class="df-modal-overlay" id="df-inspect-modal">
            <div class="df-modal-content" onclick="event.stopPropagation();">
                <button class="df-modal-close" id="df-modal-close">&times;</button>
                <div class="df-modal-header">
                    <h3 class="df-modal-title" id="df-modal-title">AI Forensic Analysis</h3>
                </div>
                <div class="df-modal-body">
                    <div class="df-modal-preview" id="df-modal-preview"></div>
                    <div class="df-modal-details">
                        <div class="df-modal-section">
                            <div class="df-modal-label">DEEPFAKE PROBABILITY</div>
                            <div class="df-score-row">
                                <span class="df-score-text" id="df-modal-score">0%</span>
                                <span style="font-size:12px; color:#94a3b8; margin-left:8px;" id="df-modal-auth-score">(Authenticity: 100%)</span>
                            </div>
                        </div>
                        <div class="df-modal-section" id="df-why-flagged-section">
                            <div class="df-modal-label">Classification Details</div>
                            <div class="df-modal-explanation" id="df-modal-exp"></div>
                        </div>
                        <div class="df-modal-section">
                            <div class="df-modal-label">Model Ensemble Breakdown:</div>
                            <div class="df-agreement-list" id="df-modal-agreement"></div>
                        </div>
                        <div id="df-misinfo-warning"></div>
                    </div>
                </div>
            </div>
        </div>
    `;
    document.body.insertAdjacentHTML('beforeend', modalHTML);

    document.getElementById('df-modal-close').addEventListener('click', closeInspectModal);
    document.getElementById('df-inspect-modal').addEventListener('click', closeInspectModal);
}

function openInspectModal(element, result) {
    injectInspectModal();
    const modal = document.getElementById('df-inspect-modal');
    const preview = document.getElementById('df-modal-preview');
    const title = document.getElementById('df-modal-title');
    const score = document.getElementById('df-modal-score');
    const authScore = document.getElementById('df-modal-auth-score');
    const exp = document.getElementById('df-modal-exp');
    const agreement = document.getElementById('df-modal-agreement');
    const misinfo = document.getElementById('df-misinfo-warning');

    preview.innerHTML = '';
    let clone;
    if (element.tagName === 'VIDEO') {
        clone = document.createElement('video');
        clone.src = element.currentSrc || element.src;
        clone.controls = true;
    } else {
        clone = document.createElement('img');
        clone.src = element.src;
    }
    preview.appendChild(clone);

    const deepfakePct = result.deepfake_percentage ?? Math.round((result.confidence || 0) * 100);
    const authPct = result.authenticity_percentage ?? (100 - deepfakePct);
    const verdict = result.verdict || (deepfakePct >= 85 ? "DEEPFAKE" : (deepfakePct >= 60 ? "SUSPICIOUS" : "AUTHENTIC"));

    if (verdict === 'DEEPFAKE' || deepfakePct >= 85) {
        const box = document.createElement('div');
        box.className = 'df-xai-box';
        box.style.top = '15%'; box.style.left = '25%'; box.style.width = '50%'; box.style.height = '50%';
        preview.appendChild(box);
    }

    animateValue(score, 0, deepfakePct, 800);
    authScore.innerText = `(Authenticity: ${authPct}%)`;

    if (verdict === 'DEEPFAKE') {
        title.innerHTML = `⚠️ Deepfake Detected`;
        score.style.color = '#ff2a4b';
        exp.innerHTML = `${result.reason || "Both models detected strong evidence of AI-generated or manipulated content."}<br><br>` +
            `The media was classified as <strong>DEEPFAKE</strong> because the ensemble detected a ${deepfakePct}% probability of AI-generated or manipulated content.`;
    } else if (verdict === 'SUSPICIOUS') {
        title.innerHTML = `🟡 Suspicious — Further Verification Recommended`;
        score.style.color = '#eab308';
        exp.innerHTML = `${result.reason || "The ensemble detected moderate evidence of AI-generated or manipulated content."}<br><br>` +
            `The media was classified as <strong>SUSPICIOUS</strong> because the ensemble detected a ${deepfakePct}% probability of AI-generated or manipulated content (exceeding the 60% threshold).`;
    } else {
        title.innerHTML = `✅ Authentic Media`;
        score.style.color = '#22c55e';
        exp.innerHTML = `${result.reason || "Low evidence of AI-generated or manipulated content detected."}<br><br>` +
            `The media was classified as <strong>AUTHENTIC</strong> because the ensemble detected low evidence of manipulation (${deepfakePct}% deepfake probability).`;
    }

    agreement.innerHTML = '';
    const detailsText = result.details || `Model 1: ${result.model1_score ? (result.model1_score * 100).toFixed(1) : 'N/A'}%, Model 2: ${result.model2_score ? (result.model2_score * 100).toFixed(1) : 'N/A'}%`;
    const detailItems = detailsText.split('. ').filter(d => d.trim());
    detailItems.forEach(d => {
        const div = document.createElement('div');
        div.className = 'df-agreement-item';
        div.innerText = d.trim() + (d.endsWith('.') ? '' : '.');
        agreement.appendChild(div);
    });

    misinfo.innerHTML = '';
    if (verdict !== 'AUTHENTIC' && checkForMisinfoKeywords()) {
        misinfo.innerHTML = `<div class="df-misinfo-tag">⚠️ Potential Misinformation Indicator Detected on Page</div>`;
    }

    modal.classList.add('active');
}

// Inject Floating Popout for Text Claim Verification
function injectTextVerificationPopout() {
    if (document.getElementById('satya-text-popout')) return;
    const popoutHTML = `
        <div class="satya-text-popout" id="satya-text-popout">
            <div class="satya-popout-header" id="satya-popout-header">
                <div class="satya-popout-title-group">
                    <div class="satya-popout-dot"></div>
                    <div class="satya-popout-title">SATYA Fact-Check</div>
                </div>
                <div class="satya-popout-controls">
                    <button class="satya-popout-btn" id="satya-popout-min" title="Minimize / Expand">_</button>
                    <button class="satya-popout-btn" id="satya-popout-close" title="Close">&times;</button>
                </div>
            </div>
            <div class="satya-popout-body" id="satya-popout-body">
                <div class="satya-claim-preview" id="satya-claim-preview"></div>
                <!-- Progress Stages -->
                <div id="satya-progress-box" style="padding: 12px 0 16px 0; text-align: center;">
                    <div class="df-loader" style="margin: 0 auto 12px auto; width: 24px; height: 24px; border-width: 2px;"></div>
                    <div id="satya-stage-label" style="font-weight: 800; font-size: 13px; color: #ff2a4b; letter-spacing: 0.5px;">STAGE 1: SCANNING</div>
                    <div class="satya-progress-bar-container">
                        <div class="satya-progress-bar-fill" id="satya-progress-bar"></div>
                    </div>
                    <div id="satya-stage-sub" style="font-size: 11px; color: #94a3b8; margin-top: 6px;">Initializing claim verification router...</div>
                </div>
                <!-- Result Card Container -->
                <div id="satya-result-container" style="display:none;"></div>
            </div>
        </div>
    `;
    document.body.insertAdjacentHTML('beforeend', popoutHTML);

    document.getElementById('satya-popout-close').addEventListener('click', closeTextPopout);
    document.getElementById('satya-popout-min').addEventListener('click', toggleMinimizePopout);
}

function closeTextPopout() {
    const popout = document.getElementById('satya-text-popout');
    if (popout) {
        popout.classList.remove('active');
        popout.classList.remove('minimized');
        const minBtn = document.getElementById('satya-popout-min');
        if (minBtn) minBtn.innerText = '_';
    }
}

function toggleMinimizePopout() {
    const popout = document.getElementById('satya-text-popout');
    const minBtn = document.getElementById('satya-popout-min');
    if (popout) {
        popout.classList.toggle('minimized');
        if (minBtn) {
            minBtn.innerText = popout.classList.contains('minimized') ? '□' : '_';
        }
    }
}

async function runTextVerification(text) {
    injectTextVerificationPopout();
    const popout = document.getElementById('satya-text-popout');
    const claimPreview = document.getElementById('satya-claim-preview');
    const progressBox = document.getElementById('satya-progress-box');
    const stageLabel = document.getElementById('satya-stage-label');
    const stageSub = document.getElementById('satya-stage-sub');
    const progressBar = document.getElementById('satya-progress-bar');
    const resultContainer = document.getElementById('satya-result-container');
    const minBtn = document.getElementById('satya-popout-min');

    // Unminimize and activate
    popout.classList.remove('minimized');
    if (minBtn) minBtn.innerText = '_';
    popout.classList.add('active');

    const cleanSnippet = text.trim();
    claimPreview.innerText = '“' + (cleanSnippet.length > 130 ? cleanSnippet.substring(0, 130) + '...' : cleanSnippet) + '”';

    progressBox.style.display = 'block';
    resultContainer.style.display = 'none';
    progressBar.style.width = '0%';

    // 7-Stage Animated Verification Sequence
    const stages = [
        { label: "STAGE 1: SCANNING", sub: "Reading input claim payload..." },
        { label: "STAGE 2: EXTRACTING CLAIM", sub: "Stripping forward markers, noise, and emotional filler..." },
        { label: "STAGE 3: SEARCHING EVIDENCE", sub: "Querying PIB, fact-checking databases, and official archives..." },
        { label: "STAGE 4: CHECKING DATES", sub: "Analyzing temporal context and publication timestamps..." },
        { label: "STAGE 5: ANALYZING SOURCES", sub: "Evaluating domain credibility and primary sources..." },
        { label: "STAGE 6: FUSING EVIDENCE", sub: "Applying multi-signal evidence fusion algorithms..." },
        { label: "STAGE 7: GENERATING VERDICT", sub: "Finalizing calibrated confidence score and explanation..." }
    ];

    for (let i = 0; i < stages.length; i++) {
        stageLabel.innerText = stages[i].label;
        stageSub.innerText = stages[i].sub;
        progressBar.style.width = `${Math.round(((i + 1) / stages.length) * 100)}%`;
        await new Promise(r => setTimeout(r, 180));
    }

    try {
        const resp = await fetch(VERIFY_TEXT_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text, target_lang: currentTargetLang })
        });

        if (!resp.ok) throw new Error("Verification service unavailable");
        const res = await resp.json();

        progressBox.style.display = 'none';
        resultContainer.style.display = 'block';

        renderUnifiedTextCard(resultContainer, res);

        chrome.runtime.sendMessage({ action: "log_scan", result: res, url: window.location.href });
        window.tabStats.textVerified++;

    } catch (err) {
        progressBox.style.display = 'none';
        resultContainer.style.display = 'block';
        resultContainer.innerHTML = `
            <div style="text-align:center; padding: 20px; color:#ff2a4b;">
                <div style="font-size:28px; margin-bottom:8px;">⚠️</div>
                <div style="font-size:13px; font-weight:800;">UNVERIFIABLE</div>
                <div style="font-size:11px; color:#94a3b8; margin-top:6px; line-height:1.4;">Verification service is unreachable or encountered an error. Please ensure the backend is running.</div>
            </div>
        `;
    }
}

function renderUnifiedTextCard(container, res) {
    const verdict = res.translated_verdict || res.verdict;
    const confidence = res.confidence || 0;
    const summary = res.summary || "Claim processed through NLI evidence pipeline.";
    const explanation = res.translated_explanation || res.explanation;

    let badgeColor = "#22c55e";
    let icon = "🟢";
    if (res.verdict === "LIKELY_FALSE") { badgeColor = "#ff2a4b"; icon = "🔴"; }
    else if (res.verdict === "UNVERIFIABLE") { badgeColor = "#eab308"; icon = "🟠"; }

    let sourcesHTML = "";
    if (res.sources && res.sources.length > 0) {
        sourcesHTML = res.sources.map((s, idx) => `
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); padding:8px 10px; border-radius:6px; margin-bottom:6px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2px;">
                    <span style="font-size:11px; font-weight:800; color:#fff;">${idx+1}. ${s.name}</span>
                    <span style="font-size:9px; background:rgba(255,255,255,0.1); padding:2px 6px; border-radius:4px; color:${s.stance === 'SUPPORTS' ? '#22c55e' : (s.stance === 'CONTRADICTS' ? '#ff2a4b' : '#eab308')}; font-weight:700;">${s.stance}</span>
                </div>
                <div style="font-size:10px; color:#cbd5e1; margin-bottom:4px; line-height:1.3;">${s.title}</div>
                <a href="${s.url}" target="_blank" rel="noopener noreferrer" style="font-size:9px; color:#60a5fa; text-decoration:none; display:inline-flex; align-items:center; gap:3px;">View Document &nearr;</a>
            </div>
        `).join('');
    } else {
        sourcesHTML = `<div style="font-size:10px; color:#64748b; padding:6px 0;">No direct third-party fact-check documents returned.</div>`;
    }

    const auditHash = res.blockchain_audit ? res.blockchain_audit.sha256_hash : "N/A";
    const isTamil = currentTargetLang === "ta";

    const viewExpLabel = isTamil ? "▶ 3-பகுதி விளக்கத்தை காண்க (3-Part Explanation)" : "▶ View 3-Part Explanation";
    const viewSrcLabel = isTamil ? `▶ ஆதாரங்களை காண்க (${res.sources ? res.sources.length : 0} Sources)` : `▶ View Evidence & Sources (${res.sources ? res.sources.length : 0})`;
    const viewAuditLabel = isTamil ? "▶ SHA-256 தணிக்கை விவரம் (Blockchain Audit)" : "▶ View Blockchain Provenance Audit";

    container.innerHTML = `
        <!-- Compact Immediately Visible Summary Card -->
        <div style="border-left: 4px solid ${badgeColor}; padding: 10px 12px; background: rgba(255,255,255,0.03); border-radius: 8px; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <div style="font-size: 15px; font-weight: 900; color: ${badgeColor}; display: flex; align-items: center; gap: 6px;">
                    <span>${icon}</span> ${verdict}
                </div>
                <div style="font-size: 11px; font-weight: 800; color: #f8fafc; background: rgba(255,255,255,0.08); padding: 3px 8px; border-radius: 6px;">
                    ${confidence}% Confidence
                </div>
            </div>
            <div style="font-size: 11px; color: #cbd5e1; font-weight: 500; line-height: 1.45;">
                ${summary}
            </div>
        </div>

        <!-- Accordion 1: 3-Part Explanation -->
        <div style="margin-bottom: 8px;">
            <button class="satya-accordion-btn" id="satya-acc-exp-btn">
                <span class="satya-acc-title" id="satya-acc-exp-title">${viewExpLabel}</span>
                <span style="opacity:0.6; font-size:10px;">☰</span>
            </button>
            <div class="satya-accordion-content" id="satya-acc-exp-panel" style="display:none; white-space:pre-wrap;">
${explanation}
            </div>
        </div>

        <!-- Accordion 2: Sources Card -->
        <div style="margin-bottom: 8px;">
            <button class="satya-accordion-btn" id="satya-acc-src-btn">
                <span class="satya-acc-title" id="satya-acc-src-title">${viewSrcLabel}</span>
                <span style="opacity:0.6; font-size:10px;">🔗</span>
            </button>
            <div class="satya-accordion-content" id="satya-acc-src-panel" style="display:none; max-height:160px; overflow-y:auto;">
                ${sourcesHTML}
            </div>
        </div>

        <!-- Accordion 3: Blockchain Audit -->
        <div style="margin-bottom: 10px;">
            <button class="satya-accordion-btn" id="satya-acc-audit-btn">
                <span class="satya-acc-title" id="satya-acc-audit-title">${viewAuditLabel}</span>
                <span style="opacity:0.6; font-size:10px;">🛡️</span>
            </button>
            <div class="satya-accordion-content" id="satya-acc-audit-panel" style="display:none; font-size:10px; color:#94a3b8;">
                <div style="color:#cbd5e1; margin-bottom:2px; font-weight:600;">SHA-256 Provenance Hash:</div>
                <code style="color:#60a5fa; word-break:break-all; font-size:9px;">${auditHash}</code>
                <div style="margin-top:6px; font-size:9px; color:#64748b;">Status: RECORDED_LOCAL_LEDGER | Engine: SATYA v2.5 Multi-Model</div>
            </div>
        </div>

        <!-- Footer Actions -->
        <div style="display:flex; justify-content:space-between; align-items:center; padding-top:4px;">
            <button class="satya-copy-btn" id="satya-copy-btn">
                📋 Copy Verdict
            </button>
            <div style="font-size:10px; color:#475569; font-weight:600;">
                SATYA AI v2.5
            </div>
        </div>
    `;

    // Attach CSP-Compliant Event Listeners (Fixes WhatsApp Web & High-Security CSP)
    const expBtn = document.getElementById('satya-acc-exp-btn');
    const expPanel = document.getElementById('satya-acc-exp-panel');
    const expTitle = document.getElementById('satya-acc-exp-title');
    if (expBtn && expPanel) {
        expBtn.addEventListener('click', () => {
            const isHidden = expPanel.style.display === 'none' || !expPanel.style.display;
            expPanel.style.display = isHidden ? 'block' : 'none';
            if (expTitle) expTitle.innerText = isHidden ? '▼ Hide Explanation' : viewExpLabel;
        });
    }

    const srcBtn = document.getElementById('satya-acc-src-btn');
    const srcPanel = document.getElementById('satya-acc-src-panel');
    const srcTitle = document.getElementById('satya-acc-src-title');
    if (srcBtn && srcPanel) {
        srcBtn.addEventListener('click', () => {
            const isHidden = srcPanel.style.display === 'none' || !srcPanel.style.display;
            srcPanel.style.display = isHidden ? 'block' : 'none';
            if (srcTitle) srcTitle.innerText = isHidden ? '▼ Hide Sources' : viewSrcLabel;
        });
    }

    const auditBtn = document.getElementById('satya-acc-audit-btn');
    const auditPanel = document.getElementById('satya-acc-audit-panel');
    const auditTitle = document.getElementById('satya-acc-audit-title');
    if (auditBtn && auditPanel) {
        auditBtn.addEventListener('click', () => {
            const isHidden = auditPanel.style.display === 'none' || !auditPanel.style.display;
            auditPanel.style.display = isHidden ? 'block' : 'none';
            if (auditTitle) auditTitle.innerText = isHidden ? '▼ Hide Audit' : viewAuditLabel;
        });
    }

    const copyBtn = document.getElementById('satya-copy-btn');
    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            const textToCopy = `SATYA Fact-Check Verdict: ${verdict} (${confidence}% Confidence)\n\nSummary: ${summary}`;
            navigator.clipboard.writeText(textToCopy).then(() => {
                copyBtn.innerHTML = '✓ Copied to Clipboard';
                setTimeout(() => { copyBtn.innerHTML = '📋 Copy Verdict'; }, 2000);
            }).catch(() => {
                copyBtn.innerHTML = '✓ Copied';
            });
        });
    }
}

async function runImageProvenanceAnalysis(srcUrl) {
    injectTextVerificationPopout();
    const popout = document.getElementById('satya-text-popout');
    const claimPreview = document.getElementById('satya-claim-preview');
    const progressBox = document.getElementById('satya-progress-box');
    const stageLabel = document.getElementById('satya-stage-label');
    const stageSub = document.getElementById('satya-stage-sub');
    const progressBar = document.getElementById('satya-progress-bar');
    const resultContainer = document.getElementById('satya-result-container');
    const minBtn = document.getElementById('satya-popout-min');

    popout.classList.remove('minimized');
    if (minBtn) minBtn.innerText = '_';
    popout.classList.add('active');

    // Find image element in page to extract alt text, title, or surrounding caption
    let extractedCaption = document.title || "";
    try {
        const imgElements = document.querySelectorAll(`img[src="${srcUrl}"]`);
        if (imgElements && imgElements.length > 0) {
            const imgEl = imgElements[0];
            const alt = imgEl.getAttribute('alt') || imgEl.getAttribute('title') || imgEl.getAttribute('aria-label') || "";
            const parentText = imgEl.closest('figure, div, a')?.innerText || "";
            if (alt || parentText) {
                extractedCaption = `${alt} ${parentText.substring(0, 100)}`.trim();
            }
        }
    } catch (e) {}

    claimPreview.innerHTML = `
        <div style="display:flex; align-items:center; gap:8px;">
            <img src="${srcUrl}" style="width:36px; height:36px; object-fit:cover; border-radius:4px; border:1px solid rgba(255,255,255,0.2);">
            <div style="font-size:11px; color:#94a3b8; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${extractedCaption ? extractedCaption.substring(0, 45) : 'Reverse Image Search & Date Analysis'}</div>
        </div>
    `;

    progressBox.style.display = 'block';
    resultContainer.style.display = 'none';
    progressBar.style.width = '0%';

    const stages = [
        { label: "STAGE 1: EXIF METADATA", sub: "Reading embedded creation timestamps and camera tags..." },
        { label: "STAGE 2: VISUAL HASHING", sub: "Computing 64-bit perceptual difference hash..." },
        { label: "STAGE 3: REVERSE SEARCH", sub: "Searching global web archives and historical indices..." },
        { label: "STAGE 4: DATE COMPARISON", sub: "Calculating temporal difference and context discrepancy..." },
        { label: "STAGE 5: PROVENANCE AUDIT", sub: "Generating visual timeline and cryptographic hash..." }
    ];

    for (let i = 0; i < stages.length; i++) {
        stageLabel.innerText = stages[i].label;
        stageSub.innerText = stages[i].sub;
        progressBar.style.width = `${Math.round(((i + 1) / stages.length) * 100)}%`;
        await new Promise(r => setTimeout(r, 40));
    }

    try {
        const formData = new FormData();
        formData.append("image_url", srcUrl);
        formData.append("claim_text", extractedCaption);
        formData.append("target_lang", currentTargetLang);

        const resp = await fetch("http://127.0.0.1:8000/analyze-image-provenance", {
            method: "POST",
            body: formData
        });

        if (!resp.ok) throw new Error("Image provenance service unavailable");
        const res = await resp.json();

        progressBox.style.display = 'none';
        resultContainer.style.display = 'block';

        renderImageProvenanceCard(resultContainer, res, srcUrl);
    } catch (err) {
        progressBox.style.display = 'none';
        resultContainer.style.display = 'block';
        resultContainer.innerHTML = `
            <div style="text-align:center; padding: 20px; color:#ff2a4b;">
                <div style="font-size:28px; margin-bottom:8px;">⚠️</div>
                <div style="font-size:13px; font-weight:800;">UNABLE TO REVERSE SEARCH</div>
                <div style="font-size:11px; color:#94a3b8; margin-top:6px; line-height:1.4;">Image provenance service was unreachable. Please ensure the backend server is running.</div>
            </div>
        `;
    }
}

function renderImageProvenanceCard(container, res, srcUrl) {
    const verdict = res.verdict || "UNKNOWN";
    const confidence = res.confidence || 0;
    const summary = res.summary || "";
    const firstYear = res.first_seen_year || new Date().getFullYear();
    const elapsed = res.years_elapsed || 0;

    let badgeColor = "#22c55e";
    let icon = "🟢";
    if (res.verdict === "OUT_OF_CONTEXT" || res.verdict === "SYNTHETIC_OR_EDITED") {
        badgeColor = "#ff2a4b";
        icon = "🔴";
    } else if (res.verdict === "ARCHIVAL_MEDIA") {
        badgeColor = "#eab308";
        icon = "🟠";
    }

    let timelineHTML = "";
    if (res.timeline && res.timeline.length > 0) {
        timelineHTML = res.timeline.map((t, idx) => `
            <div style="display:flex; gap:10px; margin-bottom:8px; position:relative;">
                <div style="font-size:11px; font-weight:800; color:#60a5fa; min-width:40px;">${t.year}</div>
                <div style="font-size:10px; color:#cbd5e1; line-height:1.35; flex:1;">
                    <span style="font-weight:700; color:#fff;">[${t.stage}]</span> ${t.description}
                </div>
            </div>
        `).join('');
    }

    const auditHash = res.blockchain_audit ? res.blockchain_audit.sha256_hash : "N/A";

    container.innerHTML = `
        <div style="border-left: 4px solid ${badgeColor}; padding: 10px 12px; background: rgba(255,255,255,0.03); border-radius: 8px; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <div style="font-size: 14px; font-weight: 900; color: ${badgeColor}; display: flex; align-items: center; gap: 6px;">
                    <span>${icon}</span> ${verdict}
                </div>
                <div style="font-size: 11px; font-weight: 800; color: #f8fafc; background: rgba(255,255,255,0.08); padding: 3px 8px; border-radius: 6px;">
                    ${confidence}%
                </div>
            </div>
            <div style="font-size: 11px; color: #cbd5e1; font-weight: 500; line-height: 1.45;">
                ${summary}
            </div>
        </div>

        <!-- Provenance Timeline Accordion -->
        <div style="margin-bottom: 8px;">
            <button class="satya-accordion-btn" id="satya-acc-timeline-btn">
                <span class="satya-acc-title" id="satya-acc-timeline-title">▶ View Provenance Timeline</span>
                <span style="opacity:0.6; font-size:10px;">⏳</span>
            </button>
            <div class="satya-accordion-content" id="satya-acc-timeline-panel" style="display:block; padding:10px; background:rgba(0,0,0,0.3); border-radius:6px;">
                ${timelineHTML}
            </div>
        </div>

        <!-- EXIF & Hash Details Accordion -->
        <div style="margin-bottom: 10px;">
            <button class="satya-accordion-btn" id="satya-acc-meta-btn">
                <span class="satya-acc-title" id="satya-acc-meta-title">▶ View Metadata & Visual Hash</span>
                <span style="opacity:0.6; font-size:10px;">🔍</span>
            </button>
            <div class="satya-accordion-content" id="satya-acc-meta-panel" style="display:none; font-size:10px; color:#94a3b8;">
                <div><strong>Perceptual Hash:</strong> <code style="color:#60a5fa;">${res.perceptual_hash || 'N/A'}</code></div>
                <div style="margin-top:4px;"><strong>Original Date:</strong> ${res.exif_metadata?.date_time_original || 'Not embedded in file'}</div>
                <div style="margin-top:4px;"><strong>Camera / Software:</strong> ${res.exif_metadata?.software || res.exif_metadata?.camera_model || 'Standard Digital Export'}</div>
                <div style="margin-top:6px; border-top:1px solid rgba(255,255,255,0.06); padding-top:6px;">
                    <div style="color:#cbd5e1;">Blockchain Provenance:</div>
                    <code style="color:#60a5fa; font-size:9px; word-break:break-all;">${auditHash}</code>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div style="display:flex; justify-content:space-between; align-items:center; padding-top:4px;">
            <button class="satya-copy-btn" id="satya-copy-img-btn">
                📋 Copy Provenance
            </button>
            <div style="font-size:10px; color:#475569; font-weight:600;">
                SATYA AI v2.5
            </div>
        </div>
    `;

    const tlBtn = document.getElementById('satya-acc-timeline-btn');
    const tlPanel = document.getElementById('satya-acc-timeline-panel');
    const tlTitle = document.getElementById('satya-acc-timeline-title');
    if (tlBtn && tlPanel) {
        tlBtn.addEventListener('click', () => {
            const isHidden = tlPanel.style.display === 'none' || !tlPanel.style.display;
            tlPanel.style.display = isHidden ? 'block' : 'none';
            if (tlTitle) tlTitle.innerText = isHidden ? '▼ Hide Provenance Timeline' : '▶ View Provenance Timeline';
        });
    }

    const metaBtn = document.getElementById('satya-acc-meta-btn');
    const metaPanel = document.getElementById('satya-acc-meta-panel');
    const metaTitle = document.getElementById('satya-acc-meta-title');
    if (metaBtn && metaPanel) {
        metaBtn.addEventListener('click', () => {
            const isHidden = metaPanel.style.display === 'none' || !metaPanel.style.display;
            metaPanel.style.display = isHidden ? 'block' : 'none';
            if (metaTitle) metaTitle.innerText = isHidden ? '▼ Hide Metadata' : '▶ View Metadata & Visual Hash';
        });
    }

    const copyImgBtn = document.getElementById('satya-copy-img-btn');
    if (copyImgBtn) {
        copyImgBtn.addEventListener('click', () => {
            const textToCopy = `SATYA Visual Provenance: ${verdict} (${confidence}% Confidence)\n\nFirst Seen: ${firstYear} (${elapsed} yrs ago)\nSummary: ${summary}`;
            navigator.clipboard.writeText(textToCopy).then(() => {
                copyImgBtn.innerHTML = '✓ Copied';
                setTimeout(() => { copyImgBtn.innerHTML = '📋 Copy Provenance'; }, 2000);
            }).catch(() => {
                copyImgBtn.innerHTML = '✓ Copied';
            });
        });
    }
}

function checkForMisinfoKeywords() {
    const text = document.body.innerText.toLowerCase();
    return FAKE_NEWS_KEYWORDS.some(kw => text.includes(kw));
}

function animateValue(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.innerHTML = Math.floor(progress * (end - start) + start) + "%";
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

function closeInspectModal() {
    const modal = document.getElementById('df-inspect-modal');
    if (modal) {
        modal.classList.remove('active');
        setTimeout(() => { document.getElementById('df-modal-preview').innerHTML = ''; }, 300);
    }
}

// Runtime messaging
chrome.runtime.sendMessage({ action: "get_status" }, (res) => {
    if (res) {
        if (res.enabled === false) extensionEnabled = false;
        if (res.targetLang) currentTargetLang = res.targetLang;
    }
    if (extensionEnabled) {
        runScan();
        setInterval(runScan, 3000);
    }
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.action === "scan_now") {
        extensionEnabled = true;
        scannedCache.clear();
        document.querySelectorAll('[data-df-scanned]').forEach(el => {
            el.removeAttribute('data-df-scanned');
            if (el.tagName === 'VIDEO' && el.dataset.dfTimer) {
                clearInterval(el.dataset.dfTimer);
                el.removeAttribute('data-df-timer');
            }
        });
        runScan();
    }
    if (msg.action === "verify_selected_text") {
        if (msg.text) runTextVerification(msg.text);
    }
    if (msg.action === "verify_image_provenance") {
        if (msg.srcUrl) runImageProvenanceAnalysis(msg.srcUrl);
    }
    if (msg.action === "get_tab_stats") {
        sendResponse(window.tabStats);
    }
});

async function checkMedia(blob, element, sourceUrl, isVideo = false) {
    if (!extensionEnabled) return;

    if (sourceUrl && scannedCache.has(sourceUrl)) {
        showBadgeAgain(element, scannedCache.get(sourceUrl));
        return;
    }

    setStatusBadge(element, "Scanning...");

    const formData = new FormData();
    formData.append("file", blob, "media.jpg");

    try {
        const response = await fetch(API_URL, { method: "POST", body: formData });
        if (!response.ok) throw new Error("Server disconnected");
        const result = await response.json();

        updateFinishedBadge(element, result);
        if (sourceUrl) scannedCache.set(sourceUrl, result);

    } catch (err) {
        setStatusBadge(element, "Analysis Failed");
    }
}

function updateVideoTimeline(video, frames) {
    let badge = getOrCreateBadge(video);
    let timeline = badge.querySelector('.df-timeline');
    if (!timeline) {
        timeline = document.createElement('div');
        timeline.className = 'df-timeline';
        badge.appendChild(timeline);
    }

    timeline.innerHTML = '';
    const segmentWidth = 100 / Math.max(frames.length, 10);
    frames.forEach((status, i) => {
        const seg = document.createElement('div');
        seg.className = 'df-timeline-segment';
        seg.style.width = segmentWidth + '%';
        seg.style.left = (i * segmentWidth) + '%';
        seg.style.backgroundColor = status === 'fake' ? '#ff2a4b' : (status === 'suspicious' ? '#eab308' : '#22c55e');
        timeline.appendChild(seg);
    });
}

function getOrCreateBadge(element) {
    let badge = badges.get(element);
    if (!badge) {
        badge = document.createElement('div');
        badge.className = "df-badge-container";

        let iconNode = document.createElement('div');
        iconNode.className = "df-badge-icon";
        badge.appendChild(iconNode);

        document.body.appendChild(badge);
        badges.set(element, badge);
    }
    return badge;
}

function setStatusBadge(element, statusText) {
    let badge = getOrCreateBadge(element);
    badge.classList.add('visible');
    const icon = badge.querySelector('.df-badge-icon');

    icon.innerHTML = `<div class="df-loader"></div> <span>Scanning...</span>`;
    icon.style.backgroundColor = "rgba(0,0,0,0.8)";

    positionBadge(element, badge);
}

function updateFinishedBadge(element, result) {
    let badge = getOrCreateBadge(element);
    const icon = badge.querySelector('.df-badge-icon');
    icon.classList.add('result-ready');

    const verdict = result.verdict || "AUTHENTIC";
    const deepfakePct = result.deepfake_percentage ?? Math.round((result.confidence || 0) * 100);

    element.classList.remove('df-border-fake', 'df-border-suspicious', 'df-border-real');

    if (verdict === 'DEEPFAKE') {
        icon.innerHTML = `<span class="df-cat-mini angry">😡</span> DEEPFAKE DETECTED (${deepfakePct}% Deepfake)`;
        icon.style.border = "1px solid var(--red-glow)";
        element.classList.add('df-border-fake');
        injectThreatBanner();
        document.getElementById('df-threat-banner').classList.add('show');
        window.tabStats.fake++;
    } else if (verdict === 'SUSPICIOUS') {
        icon.innerHTML = `<span class="df-cat-mini">🤨</span> SUSPICIOUS (${deepfakePct}% Deepfake)`;
        icon.style.border = "1px solid var(--yellow-glow)";
        element.classList.add('df-border-suspicious');
        window.tabStats.suspicious++;
    } else {
        icon.innerHTML = `<span class="df-cat-mini happy">😺</span> AUTHENTIC (${deepfakePct}% Deepfake)`;
        icon.style.border = "1px solid var(--green-glow)";
        element.classList.add('df-border-real');
    }

    window.tabStats.scanned++;

    badge.onclick = (e) => {
        e.preventDefault(); e.stopPropagation();
        openInspectModal(element, result);
    };

    positionBadge(element, badge);

    if (badge.dataset.timeoutId) clearTimeout(badge.dataset.timeoutId);
    badge.dataset.timeoutId = setTimeout(() => {
        badge.classList.remove('visible');
        badge.classList.add('fading');
        setTimeout(() => {
            badge.style.display = 'none';
        }, 500);
    }, 9000);
}

function showBadgeAgain(element, result) {
    let badge = getOrCreateBadge(element);
    badge.style.display = 'block';
    badge.classList.remove('fading');
    badge.classList.add('visible');
    updateFinishedBadge(element, result);
}

function positionBadge(element, badge) {
    const rect = element.getBoundingClientRect();
    if (rect.width < 30 || rect.height < 30 || window.getComputedStyle(element).display === 'none') {
        badge.style.display = 'none';
        return;
    }
    badge.style.display = 'block';
    badge.style.top = (window.scrollY + rect.top + 12) + 'px';
    badge.style.left = (window.scrollX + rect.left + rect.width - badge.offsetWidth - 12) + 'px';
}

function updateAllPositions() {
    for (let [element, badge] of badges.entries()) {
        positionBadge(element, badge);
    }
}

window.addEventListener("scroll", updateAllPositions);
window.addEventListener("resize", updateAllPositions);

function scanImages() {
    const images = document.querySelectorAll('img:not([data-df-scanned])');
    images.forEach(img => {
        if (img.width < 50 || img.height < 50 || img.naturalWidth === 0) return;
        img.dataset.dfScanned = "true";

        fetch(img.src)
            .then(r => r.blob())
            .then(blob => {
                if (blob.size > 10 * 1024 * 1024) return;
                checkMedia(blob, img, img.src, false);
            })
            .catch(e => { });
    });
}

function scanVideos() {
    const videos = document.querySelectorAll('video:not([data-df-scanned])');
    videos.forEach(video => {
        video.dataset.dfScanned = "true";
        if (!video.dataset.dfTimer) {
            video.dataset.dfTimer = setInterval(() => processVideoFrame(video), 2000);
        }
    });
}

function processVideoFrame(video) {
    if (!extensionEnabled) return;
    if (video.paused || video.ended || video.readyState < 2) return;
    try {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob((blob) => {
            if (blob) checkMedia(blob, video, video.currentSrc || video.src, true);
        }, "image/jpeg", 0.7);
    } catch (e) { }
}

function runScan() {
    if (!extensionEnabled) return;
    scanImages();
    scanVideos();
}
