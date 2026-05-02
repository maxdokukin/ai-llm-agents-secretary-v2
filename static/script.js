const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');

let currentAgentDiv = null;
let currentThinkingDiv = null;
let currentThinkingContentDiv = null;
let currentProgressBar = null;
let thinkingTokenCount = 0;

// Setup WebSocket connection dynamically based on the current host
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);

    if (data.type === "system") {
        const div = document.createElement('div');
        div.className = 'message tool-msg';
        div.textContent = data.content;
        chatContainer.appendChild(div);

        currentAgentDiv = null;
        if (currentProgressBar) currentProgressBar.classList.add('complete');
        currentThinkingDiv = null;

    } else if (data.type === "agent_thinking_chunk") {
        if (!currentThinkingDiv) {
            currentThinkingDiv = document.createElement('div');
            currentThinkingDiv.className = 'thinking-container';

            const headerDiv = document.createElement('div');
            headerDiv.className = 'thinking-header expanded';

            const progressWrapper = document.createElement('div');
            progressWrapper.className = 'progress-wrapper expanded';
            currentProgressBar = document.createElement('div');
            currentProgressBar.className = 'progress-bar';
            progressWrapper.appendChild(currentProgressBar);

            currentThinkingContentDiv = document.createElement('div');
            currentThinkingContentDiv.className = 'thinking-content expanded';

            headerDiv.addEventListener('click', () => {
                headerDiv.classList.toggle('expanded');
                progressWrapper.classList.toggle('expanded');
                currentThinkingContentDiv.classList.toggle('expanded');
            });

            currentThinkingDiv.appendChild(headerDiv);
            currentThinkingDiv.appendChild(progressWrapper);
            currentThinkingDiv.appendChild(currentThinkingContentDiv);
            chatContainer.appendChild(currentThinkingDiv);

            thinkingTokenCount = 0;
        }

        currentThinkingContentDiv.textContent += data.content;

        thinkingTokenCount++;
        let progressPercent = thinkingTokenCount % 100;

        if (progressPercent === 0) {
            currentProgressBar.style.transition = 'none';
            currentProgressBar.style.width = '0%';
            void currentProgressBar.offsetWidth; // Force CSS reflow
            currentProgressBar.style.transition = 'width 0.1s linear';
        } else {
            currentProgressBar.style.width = progressPercent + '%';
        }

    } else if (data.type === "agent_chunk") {
        // Lock progress bar and trigger smooth collapse
        if (currentProgressBar && !currentProgressBar.classList.contains('complete')) {
            currentProgressBar.classList.add('complete');

            const targetHeader = currentThinkingDiv.querySelector('.thinking-header');
            const targetContent = currentThinkingContentDiv;
            const targetProgress = currentThinkingDiv.querySelector('.progress-wrapper');

            // Delay before auto-fade so user sees 100% completion
            setTimeout(() => {
                if (targetHeader) targetHeader.classList.remove('expanded');
                if (targetContent) targetContent.classList.remove('expanded');
                if (targetProgress) targetProgress.classList.remove('expanded');
            }, 600);
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
        currentAgentDiv = null;
        currentThinkingDiv = null;
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

        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
});