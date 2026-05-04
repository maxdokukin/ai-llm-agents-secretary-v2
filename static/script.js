const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');
const loaderOverlay = document.getElementById('loader-overlay');
const asciiSpinner = document.getElementById('ascii-spinner');

let currentAgentDiv = null;
let currentThinkingDiv = null;
let currentThinkingContentDiv = null;
let currentProgressBar = null;
let currentEncryptionDiv = null;
let thinkingTokenCount = 0;

let scrambleInterval = null;
const SCRAMBLE_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@$%&*+-<>?~';
const SCRAMBLE_LENGTH = 45;

// --- Loading Animation Logic ---
const frames = ["|", "/", "-", "\\"];
let frameIndex = 0;
const spinnerInterval = setInterval(() => {
    frameIndex = (frameIndex + 1) % frames.length;
    asciiSpinner.textContent = frames[frameIndex];
}, 120);
// ------------------------------------

const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

function solveScramble(container) {
    if (scrambleInterval) clearInterval(scrambleInterval);
    const targetSpan = container.querySelector('.scramble-text');
    let solvedIndex = 0;

    const solveInterval = setInterval(() => {
        if (solvedIndex >= SCRAMBLE_LENGTH) {
            clearInterval(solveInterval);

            setTimeout(() => {
                container.style.opacity = "0";
                container.style.maxHeight = "0px";
                container.style.marginBottom = "0px";
                container.style.paddingLeft = "0px";
                container.style.borderLeftWidth = "0px";
                setTimeout(() => container.remove(), 500);
            }, 300);
            return;
        }

        let text = '';
        for (let i = 0; i < SCRAMBLE_LENGTH; i++) {
            if (i <= solvedIndex) {
                text += '#';
            } else {
                text += SCRAMBLE_CHARS.charAt(Math.floor(Math.random() * SCRAMBLE_CHARS.length));
            }
        }
        targetSpan.textContent = text;
        solvedIndex += 3;
    }, 30);
}

// --- Thinking Block Matrix Animations ---

function closeThinkingBlock(container) {
    const contentDiv = container.querySelector('.thinking-content');
    const targetHeader = container.querySelector('.thinking-header');
    const targetContentWrapper = container.querySelector('.thinking-content-wrapper');
    const targetProgress = container.querySelector('.progress-wrapper');

    if (!contentDiv) return;
    if (contentDiv._isAnimating) return;

    // Cache original text securely in JS memory, bypassing DOM attributes
    if (typeof contentDiv._originalText === 'undefined') {
        contentDiv._originalText = contentDiv.textContent;
    }

    const originalText = contentDiv._originalText;
    const len = originalText.length;

    if (len === 0) {
        if (targetHeader) targetHeader.classList.remove('expanded');
        if (targetContentWrapper) targetContentWrapper.classList.remove('expanded');
        if (targetProgress) targetProgress.classList.remove('expanded');
        return;
    }

    contentDiv._isAnimating = true;
    let solvedIndex = 0;
    const sweepSpeed = Math.max(Math.floor(len / 20), 5);

    const encryptInterval = setInterval(() => {
        if (solvedIndex >= len) {
            clearInterval(encryptInterval);
            contentDiv.textContent = '#'.repeat(len);

            setTimeout(() => {
                if (targetHeader) targetHeader.classList.remove('expanded');
                if (targetContentWrapper) targetContentWrapper.classList.remove('expanded');
                if (targetProgress) targetProgress.classList.remove('expanded');
                contentDiv._isAnimating = false;
            }, 300);
            return;
        }

        const solvedPart = '#'.repeat(solvedIndex);
        const remainPart = originalText.substring(solvedIndex);

        contentDiv.textContent = solvedPart + remainPart;
        solvedIndex += sweepSpeed;

        chatContainer.scrollTop = chatContainer.scrollHeight;
    }, 20);
}

function openThinkingBlockContent(container) {
    const contentDiv = container.querySelector('.thinking-content');
    const targetHeader = container.querySelector('.thinking-header');
    const targetContentWrapper = container.querySelector('.thinking-content-wrapper');
    const targetProgress = container.querySelector('.progress-wrapper');

    if (!contentDiv || typeof contentDiv._originalText === 'undefined') return;
    if (contentDiv._isAnimating) return;

    contentDiv._isAnimating = true;

    // Un-collapse instantly so the matrix sweep animation is visible
    if (targetHeader) targetHeader.classList.add('expanded');
    if (targetContentWrapper) targetContentWrapper.classList.add('expanded');
    if (targetProgress) targetProgress.classList.add('expanded');

    const originalText = contentDiv._originalText;
    const len = originalText.length;
    let solvedIndex = 0;
    const sweepSpeed = Math.max(Math.floor(len / 20), 5);

    const decryptInterval = setInterval(() => {
        if (solvedIndex >= len) {
            clearInterval(decryptInterval);
            contentDiv.textContent = originalText;
            contentDiv._isAnimating = false;
            return;
        }

        const solvedPart = originalText.substring(0, solvedIndex);
        const remainPart = '#'.repeat(len - solvedIndex);

        contentDiv.textContent = solvedPart + remainPart;
        solvedIndex += sweepSpeed;

        chatContainer.scrollTop = chatContainer.scrollHeight;
    }, 20);
}

function openThinkingBlock() {
    if (currentEncryptionDiv) {
        solveScramble(currentEncryptionDiv);
        currentEncryptionDiv = null;
    }

    currentThinkingDiv = document.createElement('div');
    currentThinkingDiv.className = 'thinking-container';

    const headerDiv = document.createElement('div');
    headerDiv.className = 'thinking-header expanded';

    const progressWrapper = document.createElement('div');
    progressWrapper.className = 'progress-wrapper expanded';
    currentProgressBar = document.createElement('div');
    currentProgressBar.className = 'progress-bar';
    progressWrapper.appendChild(currentProgressBar);

    const contentWrapper = document.createElement('div');
    contentWrapper.className = 'thinking-content-wrapper expanded';

    currentThinkingContentDiv = document.createElement('div');
    currentThinkingContentDiv.className = 'thinking-content';

    contentWrapper.appendChild(currentThinkingContentDiv);

    headerDiv.addEventListener('click', function() {
        const container = headerDiv.parentElement;
        const myContentDiv = container.querySelector('.thinking-content');

        // If still generating (no cached text yet), standard CSS toggle
        if (typeof myContentDiv._originalText === 'undefined') {
            headerDiv.classList.toggle('expanded');
            progressWrapper.classList.toggle('expanded');
            contentWrapper.classList.toggle('expanded');
            return;
        }

        // If completed, trigger matrix encryption/decryption animations
        if (headerDiv.classList.contains('expanded')) {
            closeThinkingBlock(container);
        } else {
            openThinkingBlockContent(container);
        }
    });

    currentThinkingDiv.appendChild(headerDiv);
    currentThinkingDiv.appendChild(progressWrapper);
    currentThinkingDiv.appendChild(contentWrapper);
    chatContainer.appendChild(currentThinkingDiv);

    thinkingTokenCount = 0;
}

// -----------------------------------------------------------

function updateThinkingProgress() {
    thinkingTokenCount++;
    let progressPercent = thinkingTokenCount % 100;

    if (progressPercent === 0) {
        currentProgressBar.style.transition = 'none';
        currentProgressBar.style.width = '0%';
        void currentProgressBar.offsetWidth;
        currentProgressBar.style.transition = 'width 0.1s linear';
    } else {
        currentProgressBar.style.width = progressPercent + '%';
    }
}

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);

    if (data.type === "system") {

        if (data.content.includes("[CONNECTED]") && !loaderOverlay.classList.contains('hidden')) {
            clearInterval(spinnerInterval);
            loaderOverlay.classList.add('hidden');
            userInput.disabled = false;
            userInput.placeholder = "Awaiting command...";
            userInput.focus();
        }

        const div = document.createElement('div');
        div.className = 'message tool-msg';
        div.textContent = data.content;
        chatContainer.appendChild(div);

        currentAgentDiv = null;

        if (currentProgressBar && !currentProgressBar.classList.contains('complete')) {
            currentProgressBar.classList.add('complete');
            if (currentThinkingDiv) closeThinkingBlock(currentThinkingDiv);
        }

        currentThinkingDiv = null;
        currentProgressBar = null;

    } else if (data.type === "agent_thinking_chunk") {

        if (!currentThinkingDiv) openThinkingBlock();
        currentThinkingContentDiv.textContent += data.content;
        updateThinkingProgress();

    } else if (data.type === "agent_chunk") {

        if (currentProgressBar && !currentProgressBar.classList.contains('complete')) {
            currentProgressBar.classList.add('complete');
            closeThinkingBlock(currentThinkingDiv);
            currentThinkingDiv = null;
            currentProgressBar = null;
        }

        if (!currentAgentDiv) {
            currentAgentDiv = document.createElement('div');
            currentAgentDiv.className = 'message agent-msg';
            chatContainer.appendChild(currentAgentDiv);
        }
        currentAgentDiv.textContent += data.content;

    } else if (data.type === "stats") {
        const div = document.createElement('div');
        div.className = 'message stats-msg';
        div.textContent = data.content;
        chatContainer.appendChild(div);

    } else if (data.type === "done") {
        if (currentProgressBar && !currentProgressBar.classList.contains('complete')) {
            currentProgressBar.classList.add('complete');
            if (currentThinkingDiv) closeThinkingBlock(currentThinkingDiv);
        }

        currentAgentDiv = null;
        currentThinkingDiv = null;
        currentProgressBar = null;
        userInput.disabled = false;
        userInput.focus();
    }

    chatContainer.scrollTop = chatContainer.scrollHeight;
};

userInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && userInput.value.trim() !== '') {
        const text = userInput.value.trim();

        const userDiv = document.createElement('div');
        userDiv.className = 'message user-msg';
        userDiv.textContent = text;
        chatContainer.appendChild(userDiv);

        ws.send(text);

        userInput.value = '';
        userInput.disabled = true;
        currentAgentDiv = null;
        currentThinkingDiv = null;

        if (currentProgressBar) {
            currentProgressBar.classList.add('complete');
            currentProgressBar = null;
        }

        currentEncryptionDiv = document.createElement('div');
        currentEncryptionDiv.className = 'encryption-container';

        const header = document.createElement('div');
        header.className = 'encryption-header';

        const label = document.createElement('span');
        label.className = 'encryption-label';

        const scrambleText = document.createElement('span');
        scrambleText.className = 'scramble-text';

        header.appendChild(label);
        header.appendChild(scrambleText);
        currentEncryptionDiv.appendChild(header);
        chatContainer.appendChild(currentEncryptionDiv);

        if (scrambleInterval) clearInterval(scrambleInterval);
        scrambleInterval = setInterval(() => {
            let rndText = '';
            for (let i = 0; i < SCRAMBLE_LENGTH; i++) {
                rndText += SCRAMBLE_CHARS.charAt(Math.floor(Math.random() * SCRAMBLE_CHARS.length));
            }
            scrambleText.textContent = rndText;
        }, 50);

        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
});