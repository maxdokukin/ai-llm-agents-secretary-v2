const SESSION_ID = window.CONTEXT_DASHBOARD.sessionId;

function safePercent(value, max) {
    if (!max || max <= 0) {
        return 0;
    }

    return Math.max(0, Math.min(100, (value / max) * 100));
}

function setSegment(id, chars, max) {
    const el = document.getElementById(id);
    const charCount = Number(chars || 0);
    const pct = safePercent(charCount, max);

    el.textContent = "";

    if (charCount <= 0) {
        el.style.width = "0%";
        el.classList.add("is-empty");
        el.classList.remove("is-visible-used");
        return;
    }

    el.style.width = pct + "%";
    el.classList.remove("is-empty");

    if (id === "seg-free") {
        el.classList.remove("is-visible-used");
    } else {
        el.classList.add("is-visible-used");
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
        const counts = usage.counts || {};

        const master = counts.master || 0;
        const tools = counts.tools || 0;
        const results = counts.results || 0;
        const index = counts.index || 0;
        const data = counts.data || 0;
        const user = counts.user || 0;
        const assistant = counts.assistant || 0;

        const used = usage.used || 0;
        const free = usage.free || Math.max(0, max - used);

        setSegment("seg-master", master, max);
        setSegment("seg-tools", tools, max);
        setSegment("seg-results", results, max);
        setSegment("seg-index", index, max);
        setSegment("seg-data", data, max);
        setSegment("seg-user", user, max);
        setSegment("seg-assistant", assistant, max);
        setSegment("seg-free", free, max);

        setText("used-count", used);
        setText("free-count", free);
        setText("total-count", max);

        setText("count-master", master);
        setText("count-tools", tools);
        setText("count-results", results);
        setText("count-index", index);
        setText("count-data", data);
        setText("count-user", user);
        setText("count-assistant", assistant);

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