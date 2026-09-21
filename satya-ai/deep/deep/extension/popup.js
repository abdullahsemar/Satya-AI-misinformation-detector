document.addEventListener("DOMContentLoaded", () => {

    const toggle = document.getElementById("scanToggle");
    const statusText = document.getElementById("scanStatusText");
    const trustVal = document.getElementById("trustScore");
    const langSelect = document.getElementById("langSelect");
    const claimInput = document.getElementById("claimInput");
    const verifyTextBtn = document.getElementById("verifyTextBtn");

    toggle.addEventListener("change", (e) => {
        const enabled = e.target.checked;
        statusText.innerText = enabled ? "ACTIVE" : "PAUSED";
        statusText.style.color = enabled ? "#ff2a4b" : "#64748b";
        chrome.runtime.sendMessage({ action: "toggle_scan", enabled });
    });

    langSelect.addEventListener("change", (e) => {
        const targetLang = e.target.value;
        chrome.runtime.sendMessage({ action: "set_language", targetLang });
    });

    verifyTextBtn.addEventListener("click", () => {
        const text = claimInput.value.trim();
        if (!text) return;

        verifyTextBtn.disabled = true;
        verifyTextBtn.innerHTML = '<span>⏳</span> Verifying Claim...';

        chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
            if (tabs[0] && !tabs[0].url.startsWith('chrome://')) {
                const tabId = tabs[0].id;
                chrome.tabs.sendMessage(tabId, {
                    action: "verify_selected_text",
                    text: text
                }, () => {
                    if (chrome.runtime.lastError) {
                        chrome.scripting.executeScript({
                            target: { tabId: tabId },
                            files: ["content.js"]
                        }, () => {
                            chrome.scripting.insertCSS({
                                target: { tabId: tabId },
                                files: ["styles.css"]
                            }, () => {
                                chrome.tabs.sendMessage(tabId, {
                                    action: "verify_selected_text",
                                    text: text
                                });
                            });
                        });
                    }
                });
            }
            setTimeout(() => {
                verifyTextBtn.disabled = false;
                verifyTextBtn.innerHTML = '<span>🔍</span> Verify Claim';
                claimInput.value = '';
                refreshData();
            }, 1000);
        });
    });

    function refreshData() {
        chrome.runtime.sendMessage({ action: "get_analytics" }, (data) => {
            if (data) {
                toggle.checked = data.enabled !== false;
                statusText.innerText = toggle.checked ? "ACTIVE" : "PAUSED";
                statusText.style.color = toggle.checked ? "#ff2a4b" : "#64748b";
                if (data.targetLang) langSelect.value = data.targetLang;

                const list = document.getElementById("historyList");
                list.innerHTML = '';
                if (data.history && data.history.length > 0) {
                    data.history.slice(0, 10).forEach(item => {
                        let state = 'authentic';
                        let label = item.verdict || 'AUTHENTIC';
                        let icon = '✅';
                        let scoreText = `${Math.round((item.confidence || 0) * (item.confidence > 1 ? 1 : 100))}%`;

                        if (item.input_type === 'text') {
                            if (label === 'LIKELY_FALSE') { state = 'fake'; icon = '🔴'; label = 'LIKELY FALSE'; }
                            else if (label === 'UNVERIFIABLE') { state = 'suspicious'; icon = '🟠'; label = 'UNVERIFIABLE'; }
                            else { state = 'authentic'; icon = '🟢'; label = 'LIKELY TRUE'; }
                        } else {
                            const deepfakePct = Math.round((item.confidence || 0) * (item.confidence > 1 ? 1 : 100));
                            if (deepfakePct >= 85) { state = 'fake'; label = 'DEEPFAKE'; icon = '⚠️'; }
                            else if (deepfakePct >= 60) { state = 'suspicious'; label = 'SUSPICIOUS'; icon = '🟡'; }
                            else { state = 'authentic'; label = 'AUTHENTIC'; icon = '✅'; }
                            scoreText = `${deepfakePct}% Deepfake`;
                        }

                        let div = document.createElement("div");
                        div.className = `history-item ${state.toLowerCase()}`;
                        div.innerHTML = `
                            <span>${icon} ${label}</span>
                            <span style="font-weight:900;">${scoreText}</span>
                        `;
                        list.appendChild(div);
                    });
                } else {
                    list.innerHTML = `<div style="text-align:center; color:#334155; font-size:10px; padding: 10px;">STANDBY: NO RECENT VERIFICATIONS</div>`;
                }
            }
        });

        chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
            if (tabs[0] && !tabs[0].url.startsWith('chrome://')) {
                chrome.tabs.sendMessage(tabs[0].id, { action: "get_tab_stats" }, (res) => {
                    if (res) updateUI(res.scanned || 0, res.fake || 0, res.suspicious || 0);
                });
            }
        });
    }

    function updateUI(scanned, fake, suspicious) {
        animateValue(document.getElementById("tabScanned"), 0, scanned, 500);
        animateValue(document.getElementById("tabFake"), 0, fake, 500);

        if (fake > 0) document.getElementById("tabFake").style.color = "#ff2a4b";

        let score = 100 - (fake * 35) - (suspicious * 10);
        score = Math.max(0, score);

        animateValue(trustVal, 100, score, 1000, true);

        const catPanel = document.getElementById("catPanel");
        const catIcon = document.getElementById("catIcon");
        const catMessage = document.getElementById("catMessage");

        if (score < 50) {
            catPanel.style.borderColor = "rgba(255, 42, 75, 0.2)";
            catPanel.style.background = "rgba(255, 42, 75, 0.05)";
            catIcon.innerText = "🙀";
            catMessage.innerText = "CRITICAL: Multiple synthetic signatures or false claims detected.";
        } else if (score < 90) {
            catPanel.style.borderColor = "rgba(234, 179, 8, 0.2)";
            catPanel.style.background = "rgba(234, 179, 8, 0.05)";
            catIcon.innerText = "🧐";
            catMessage.innerText = "Caution: Some media sources or claims exhibit suspicious characteristics.";
        } else {
            catPanel.style.borderColor = "rgba(255, 255, 255, 0.05)";
            catPanel.style.background = "#0a0a0a";
            catIcon.innerText = "😺";
            catMessage.innerText = "System operational. No significant synthetic anomalies detected.";
        }
    }

    function animateValue(obj, start, end, duration, isPct = false) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            let val = Math.floor(progress * (end - start) + start);
            obj.innerHTML = isPct ? val + "%" : val;
            if (progress < 1) window.requestAnimationFrame(step);
        };
        window.requestAnimationFrame(step);
    }

    document.getElementById('scanBtn').addEventListener('click', () => {
        const btn = document.getElementById('scanBtn');
        btn.style.pointerEvents = 'none';
        btn.style.opacity = '0.5';
        btn.innerHTML = '<span>⏳</span> REANALYZING...';

        chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
            if (tabs[0] && !tabs[0].url.startsWith('chrome://')) {
                chrome.tabs.sendMessage(tabs[0].id, { action: "scan_now" });
                setTimeout(() => {
                    btn.style.pointerEvents = 'auto';
                    btn.style.opacity = '1';
                    btn.innerHTML = '<span>🔄</span> RESCAN PAGE MEDIA';
                    refreshData();
                }, 1500);
            }
        });
    });

    refreshData();
    setInterval(refreshData, 3000);
});
