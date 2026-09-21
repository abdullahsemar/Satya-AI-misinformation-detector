// Context menu initialization
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "satya_verify_text",
        title: "🔍 Verify claim with SATYA",
        contexts: ["selection"]
    });
    chrome.contextMenus.create({
        id: "satya_verify_image_provenance",
        title: "🖼️ Reverse Search & Date Check with SATYA",
        contexts: ["image"]
    });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (!tab || !tab.id) return;

    if (info.menuItemId === "satya_verify_text" && info.selectionText) {
        chrome.tabs.sendMessage(tab.id, {
            action: "verify_selected_text",
            text: info.selectionText
        }, () => {
            if (chrome.runtime.lastError) {
                chrome.scripting.executeScript({
                    target: { tabId: tab.id },
                    files: ["content.js"]
                }, () => {
                    chrome.scripting.insertCSS({
                        target: { tabId: tab.id },
                        files: ["styles.css"]
                    }, () => {
                        chrome.tabs.sendMessage(tab.id, {
                            action: "verify_selected_text",
                            text: info.selectionText
                        });
                    });
                });
            }
        });
    }

    if (info.menuItemId === "satya_verify_image_provenance" && info.srcUrl) {
        chrome.tabs.sendMessage(tab.id, {
            action: "verify_image_provenance",
            srcUrl: info.srcUrl
        }, () => {
            if (chrome.runtime.lastError) {
                chrome.scripting.executeScript({
                    target: { tabId: tab.id },
                    files: ["content.js"]
                }, () => {
                    chrome.scripting.insertCSS({
                        target: { tabId: tab.id },
                        files: ["styles.css"]
                    }, () => {
                        chrome.tabs.sendMessage(tab.id, {
                            action: "verify_image_provenance",
                            srcUrl: info.srcUrl
                        });
                    });
                });
            }
        });
    }
});

chrome.storage.local.get({
    totalScanned: 0,
    totalFake: 0,
    history: [],
    enabled: true,
    targetLang: "en"
}, () => {});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "show_notification") {
        if (message.confidence >= 75) {
            chrome.notifications.create({
                type: "basic",
                iconUrl: "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiI+PGNpcmNsZSBjeD0iMTYiIGN5PSIxNiIgcj0iMTYiIGZpbGw9InJlZCIvPjwvc3ZnPg==",
                title: "⚠️ SATYA Security Alert",
                message: `${message.title || 'High Risk Media Detected'}: ${message.confidence}%`,
                priority: 2
            });
        }
    }

    if (message.action === "get_status") {
        chrome.storage.local.get(["enabled", "targetLang"], (res) => sendResponse({
            enabled: res.enabled !== false,
            targetLang: res.targetLang || "en"
        }));
        return true;
    }

    if (message.action === "toggle_scan") {
         chrome.storage.local.set({ enabled: message.enabled });
    }

    if (message.action === "set_language") {
        chrome.storage.local.set({ targetLang: message.targetLang });
    }

    if (message.action === "log_scan") {
        chrome.storage.local.get(["totalScanned", "totalFake", "history"], (data) => {
            let totalScanned = (data.totalScanned || 0) + 1;
            let totalFake = data.totalFake || 0;
            let history = data.history || [];
            
            if (message.result.is_deepfake || message.result.verdict === "LIKELY_FALSE") {
                totalFake++;
            }
            
            history.unshift({
                url: message.url || "Web Selection",
                input_type: message.result.input_type || "image",
                verdict: message.result.verdict,
                claim: message.result.claim || null,
                confidence: message.result.confidence,
                timestamp: Date.now()
            });
            
            if (history.length > 100) history.pop();
            
            chrome.storage.local.set({ totalScanned, totalFake, history });
            
            if (sendResponse) sendResponse({ success: true });
        });
        return true;
    }
    
    if (message.action === "get_analytics") {
        chrome.storage.local.get(["totalScanned", "totalFake", "history", "enabled", "targetLang"], (data) => {
            sendResponse(data);
        });
        return true;
    }
});
