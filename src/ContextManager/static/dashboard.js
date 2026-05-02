const SESSION_ID = window.CONTEXT_DASHBOARD.sessionId;

function safePercent(value, max) {
    if (!max || max <= 0) {
        return 0;
    }

    return Math.max(0, Math.min(100, (value / max) * 100));
}

function setSegment(id, chars, max) {
    const el = document.getElementById(id);
    const pct = safePercent(chars, max);

    el.style.width = pct + "%";
    el.textContent = "";

    if (chars <= 0 || pct < 1.25) {
        el.classList.add("is-empty");
    } else {
        el.classList.remove("is-empty");
    }
}

function setText(id, value) {
    document.getElementById(id).innerText = Number(value || 0).toLocaleString();
}

async function updateDashboard() {
    try {
        const usageRes = await fetch(
            "/api/context/usage/" + encodeURIComponent(SESSION_ID)
        );
        const usage = await usageRes.json();

        const contextRes = await fetch(
            "/api/context/assemble/" + encodeURIComponent(SESSION_ID)
        );
        const contextData = await contextRes.json();

        const messages = contextData.messages || [];

        const max = usage.max || 32768;
        const used = usage.used || 0;
        const free = usage.free || Math.max(0, max - used);
        const counts = usage.counts || {};

        setSegment("seg-master", counts.master || 0, max);
        setSegment("seg-tools", counts.tools || 0, max);
        setSegment("seg-results", counts.results || 0, max);
        setSegment("seg-index", counts.index || 0, max);
        setSegment("seg-data", counts.data || 0, max);
        setSegment("seg-history", counts.history || 0, max);
        setSegment("seg-free", free, max);

        setText("used-count", used);
        setText("free-count", free);
        setText("total-count", max);

        setText("count-master", counts.master || 0);
        setText("count-tools", counts.tools || 0);
        setText("count-results", counts.results || 0);
        setText("count-index", counts.index || 0);
        setText("count-data", counts.data || 0);
        setText("count-history", counts.history || 0);

        document.getElementById("context-json").innerText = JSON.stringify(
            messages,
            null,
            2
        );

        const badge = document.getElementById("status-badge");
        badge.innerText = "Live";
        badge.style.background = "#e5e7eb";
        badge.style.color = "#111827";
    } catch (e) {
        console.error("Update failed", e);

        const badge = document.getElementById("status-badge");
        badge.innerText = "Disconnected";
        badge.style.background = "#fee2e2";
        badge.style.color = "#991b1b";
    }
}

setInterval(updateDashboard, 2000);
updateDashboard();