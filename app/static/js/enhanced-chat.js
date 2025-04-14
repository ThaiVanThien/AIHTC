/**
 * Enhanced Chat Interface JavaScript
 * Provides interactive functionality for the AI chat interface
 */

document.addEventListener('DOMContentLoaded', function() {
    // UI Elements
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const messagesContainer = document.getElementById('messages-wrap');
    const providerButtons = document.querySelectorAll('.provider-btn');
    const newChatButton = document.getElementById('new-chat-btn');
    const clearChatButton = document.getElementById('clear-chat-btn');
    const clearHistoryButton = document.getElementById('clear-history-btn');
    const chatThreads = document.querySelector('.chat-history');
    
    // State variables
    let currentProvider = 'vimrc'; // Default provider
    let currentModel = null; // Will be set based on provider
    let messages = []; // Store conversation history
    let currentThreadId = generateThreadId();
    let threads = {}; // Store all chat threads
    let modelConfigs = {}; // Will store models fetched from server
    
    // Initialize the chat interface
    initChat();
    
    async function initChat() {
        // Load models first
        await loadModels();
        
        // Enable/disable send button based on input
        messageInput.addEventListener('input', function() {
            sendButton.disabled = !this.value.trim();
        });

        // Handle Enter key
        messageInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (this.value.trim()) {
                    handleChatSubmit(e);
                }
            }
        });

        // Handle form submission
        chatForm.addEventListener('submit', function(e) {
            e.preventDefault();
            handleChatSubmit(e);
        });

        // Auto-resize textarea
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });

        // New Chat button
        newChatButton.addEventListener('click', createNewThread);

        // Clear current chat
        clearChatButton.addEventListener('click', clearCurrentChat);

        // Clear all history
        clearHistoryButton.addEventListener('click', clearAllHistory);

        // Initialize provider buttons
        providerButtons.forEach(btn => {
            btn.addEventListener('click', async () => {
                // Update UI
                providerButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                // Update state
                currentProvider = btn.getAttribute('data-provider');
                await updateProviderInfo();
            });
        });
        
        // Set default provider active and load initial data
        document.querySelector(`.provider-btn[data-provider="${currentProvider}"]`).classList.add('active');
        await updateProviderInfo();
        loadChatHistory();
    }
    
    async function loadModels() {
        try {
            const response = await fetch('/api/v1/chat/models');
            if (!response.ok) {
                throw new Error('Failed to load models');
            }

            const data = await response.json();
            
            // Transform the data into our expected format
            modelConfigs = {
                vimrc: {
                    models: data.vimrc.models.map(name => ({
                        name: name,
                        display_name: name
                    })),
                    default_model: data.vimrc.default
                },
                openai: {
                    models: data.openai.models.map(name => ({
                        name: name,
                        display_name: name
                    })),
                    default_model: data.openai.default
                },
                gemini: {
                    models: data.gemini.models.map(name => ({
                        name: name,
                        display_name: name
                    })),
                    default_model: data.gemini.default
                }
            };

            // Update model options for current provider
            updateModelOptions(currentProvider);

            // Set default model
            const providerConfig = modelConfigs[currentProvider];
            if (providerConfig && providerConfig.models.length > 0) {
                currentModel = providerConfig.default_model;
                const modelSelect = document.getElementById('model-select');
                if (modelSelect) {
                    modelSelect.value = currentModel;
                }
            }
        } catch (error) {
            console.error('Error loading models:', error);
            // Set some default models in case of API failure
            modelConfigs = {
                vimrc: {
                    models: [
                        { name: 'vi-mrc-large', display_name: 'VI-MRC Large' }
                    ],
                    default_model: 'vi-mrc-large'
                }
            };
            updateModelOptions(currentProvider);
        }
    }
    
    async function updateProviderInfo() {
        const botNameElement = document.querySelector('.chat-title');
        if (botNameElement) {
            botNameElement.textContent = `${currentProvider.toUpperCase()} Assistant`;
        }
        
        // Update model options for the new provider
        const config = modelConfigs[currentProvider];
        if (config && Array.isArray(config.models)) {
            updateModelOptions(currentProvider);
            
            // Set default model for new provider
            currentModel = config.default_model || (config.models.length > 0 ? config.models[0].name : null);
            
            const modelSelect = document.getElementById('model-select');
            if (modelSelect && currentModel) {
                modelSelect.value = currentModel;
            }
        } else {
            // If models haven't been loaded yet, load them
            await loadModels();
        }
    }
    
    function loadChatHistory() {
        const savedThreads = localStorage.getItem('chatThreads');
        if (savedThreads) {
            threads = JSON.parse(savedThreads);
            
            // Create thread entries in sidebar
            for (const threadId in threads) {
                addThreadToSidebar(threadId, threads[threadId].title);
            }
            
            // Load most recent thread if available
            if (Object.keys(threads).length > 0) {
                currentThreadId = Object.keys(threads)[0];
                loadThread(currentThreadId);
            }
        }
    }
    
    function loadThread(threadId) {
        if (threads[threadId]) {
            messages = threads[threadId].messages;
            currentThreadId = threadId;
            
            // Update UI
            renderMessages();
            
            // Update active thread in sidebar
            document.querySelectorAll('.thread-item').forEach(item => {
                item.classList.remove('active');
            });
            document.querySelector(`.thread-item[data-thread-id="${threadId}"]`).classList.add('active');
        }
    }
    
    function addThreadToSidebar(threadId, title) {
        const threadItem = document.createElement('div');
        threadItem.className = 'thread-item';
        threadItem.setAttribute('data-thread-id', threadId);
        
        // Create thread content wrapper
        const threadContent = document.createElement('div');
        threadContent.className = 'thread-content';
        threadContent.textContent = title || 'New Chat';
        
        // Create delete button
        const deleteButton = document.createElement('button');
        deleteButton.className = 'thread-delete-btn';
        deleteButton.innerHTML = '<i class="fas fa-times"></i>';
        deleteButton.title = 'Delete this chat';
        
        // Add click event for thread selection
        threadContent.addEventListener('click', () => {
            loadThread(threadId);
        });
        
        // Add click event for delete button
        deleteButton.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent thread selection when clicking delete
            deleteThread(threadId);
        });
        
        // Assemble thread item
        threadItem.appendChild(threadContent);
        threadItem.appendChild(deleteButton);
        
        // Add to chat history
        const chatThreads = document.querySelector('.chat-history');
        if (chatThreads) {
            chatThreads.appendChild(threadItem);
        }
    }
    
    function deleteThread(threadId) {
        if (!confirm('Are you sure you want to delete this chat?')) return;
        
        // Remove thread from storage
        delete threads[threadId];
        saveThreads();
        
        // Remove thread from UI
        const threadElement = document.querySelector(`.thread-item[data-thread-id="${threadId}"]`);
        if (threadElement) {
            threadElement.remove();
        }
        
        // If deleted current thread, create new one
        if (threadId === currentThreadId) {
            createNewThread();
        }
        // If there are other threads, load the first one
        else if (Object.keys(threads).length > 0) {
            loadThread(Object.keys(threads)[0]);
        }
    }
    
    function createNewThread() {
        // Clear current messages
        messages = [];
        currentThreadId = generateThreadId();
        
        // Clear message container
        messagesContainer.innerHTML = '';
        
        // Create welcome message
        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'message assistant-message';
        welcomeDiv.innerHTML = `
            <div class="message-content">
                <h3>Welcome to AI Chat Hub</h3>
                <p>How can I help you today?</p>
            </div>
        `;
        messagesContainer.appendChild(welcomeDiv);
        
        // Create new thread entry
        const title = 'New Chat';
        threads[currentThreadId] = {
            title: title,
            messages: messages,
            timestamp: new Date().toISOString()
        };
        
        addThreadToSidebar(currentThreadId, title);
        saveThreads();
        
        // Clear input
        messageInput.value = '';
        messageInput.style.height = 'auto';
        sendButton.disabled = true;
    }
    
    function clearCurrentChat() {
        if (!confirm('Are you sure you want to clear the current chat?')) return;

        // Clear only current thread messages
        messages = [];
        messagesContainer.innerHTML = '';
        
        // Update thread
        if (threads[currentThreadId]) {
            threads[currentThreadId].messages = [];
            saveThreads();
        }
        
        // Show empty state message
        const emptyDiv = document.createElement('div');
        emptyDiv.className = 'message assistant-message';
        emptyDiv.innerHTML = `
            <div class="message-content">
                <p>Chat cleared. Start a new conversation!</p>
            </div>
        `;
        messagesContainer.appendChild(emptyDiv);
        
        // Clear input
        messageInput.value = '';
        messageInput.style.height = 'auto';
        sendButton.disabled = true;
    }

    function clearAllHistory() {
        if (!confirm('Are you sure you want to clear all chat history? This cannot be undone.')) return;

        // Clear all threads
        threads = {};
        localStorage.removeItem('chatThreads');
        
        // Clear current messages
        messages = [];
        messagesContainer.innerHTML = '';
        
        // Clear chat history UI
        while (chatThreads.firstChild) {
            chatThreads.removeChild(chatThreads.firstChild);
        }
        
        // Create new thread
        createNewThread();
    }
    
    function generateThreadId() {
        return 'thread_' + Date.now();
    }
    
    function saveThreads() {
        // Update current thread with latest messages
        if (threads[currentThreadId]) {
            threads[currentThreadId].messages = messages;
        } else {
            threads[currentThreadId] = {
                title: messages.length > 0 ? messages[0].content.substring(0, 30) + '...' : 'New Chat',
                messages: messages,
                timestamp: new Date().toISOString()
            };
        }
        
        localStorage.setItem('chatThreads', JSON.stringify(threads));
    }
    
    async function handleChatSubmit(e) {
        e.preventDefault();
        
        const message = messageInput.value.trim();
        if (!message) return;
        
        // Clear input and reset height
        messageInput.value = '';
        messageInput.style.height = 'auto';
        sendButton.disabled = true;
        
        // Add user message
        addMessage('user', message);
        
        // Update thread title if first message
        if (messages.length === 1) {
            const threadTitle = message.length > 30 ? message.substring(0, 30) + '...' : message;
            if (threads[currentThreadId]) {
                threads[currentThreadId].title = threadTitle;
            }
            
            const threadElement = document.querySelector(`.thread-item[data-thread-id="${currentThreadId}"]`);
            if (threadElement) {
                threadElement.textContent = threadTitle;
            }
            
            saveThreads();
        }
        
        // Show loading indicator
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message assistant-message loading';
        loadingDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
        messagesContainer.appendChild(loadingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        try {
            const apiMessages = messages.map(msg => ({
                role: msg.role,
                content: msg.content,
                context: msg.context || null
            }));
            
            const response = await fetch('/api/v1/chat/send', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    messages: apiMessages,
                    provider: currentProvider,
                    model: currentModel,
                    temperature: 0.7,
                    max_tokens: 800
                })
            });
            
            messagesContainer.removeChild(loadingDiv);
            
            if (response.ok) {
                const data = await response.json();
                addMessage('assistant', data.content);
                Prism.highlightAll();
            } else {
                const errorData = await response.json();
                addMessage('assistant', `Error: ${errorData.detail || 'Failed to get response'}`);
            }
        } catch (error) {
            if (messagesContainer.contains(loadingDiv)) {
                messagesContainer.removeChild(loadingDiv);
            }
            console.error('Error sending message:', error);
            addMessage('assistant', `Error: ${error.message || 'Something went wrong'}`);
        }
    }
    
    function addMessage(role, content, context = null) {
        // Add to messages array
        messages.push({ role, content, context });
        
        // Save to localStorage
        saveThreads();
        
        // Update UI
        renderMessage({ role, content });
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    function renderMessages() {
        // Clear messages container
        messagesContainer.innerHTML = '';
        
        // If no messages, show welcome message
        if (messages.length === 0) {
            const welcomeDiv = document.createElement('div');
            welcomeDiv.className = 'welcome-message';
            welcomeDiv.innerHTML = `
                <h3>Welcome to VI-MRC Assistant</h3>
                <p>Ask me anything about Vietnamese language, context-based questions, or general knowledge.</p>
            `;
            messagesContainer.appendChild(welcomeDiv);
            return;
        }
        
        // Render all messages
        messages.forEach(message => {
            renderMessage(message);
        });
        
        // Apply syntax highlighting
        Prism.highlightAll();
    }
    
    function renderMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${message.role}-message`;
        
        let displayName = message.role === 'user' ? 'You' : 'Assistant';
        
        // Format message content - detect and format code blocks
        let formattedContent = formatMessageContent(message.content);
        
        messageDiv.innerHTML = `
            <div class="message-header">
                <span class="message-sender">${displayName}</span>
                <span class="message-time">${new Date().toLocaleTimeString()}</span>
            </div>
            <div class="message-content">${formattedContent}</div>
        `;
        
        messagesContainer.appendChild(messageDiv);
    }
    
    function formatMessageContent(content) {
        // Replace code blocks with properly formatted HTML
        let formattedContent = content;
        
        // Match Markdown style code blocks
        formattedContent = formattedContent.replace(/```([\w-]+)?\n([\s\S]*?)\n```/g, (match, language, code) => {
            const lang = language || 'plaintext';
            return `<pre><code class="language-${lang}">${escapeHtml(code)}</code></pre>`;
        });
        
        // Match inline code
        formattedContent = formattedContent.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Convert line breaks to <br>
        formattedContent = formattedContent.replace(/\n/g, '<br>');
        
        return formattedContent;
    }
    
    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Function to update model options based on selected provider
    function updateModelOptions(provider) {
        const modelSelect = document.getElementById('model-select');
        if (!modelSelect) return;
        
        modelSelect.innerHTML = ''; // Clear existing options
        
        const config = modelConfigs[provider];
        if (!config || !Array.isArray(config.models)) {
            console.error('Invalid configuration for provider:', provider);
            return;
        }
        
        config.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.name;
            // Format display name to be more readable
            option.textContent = model.display_name
                .split(/[-_]/)
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ');
            modelSelect.appendChild(option);
        });
        
        // Set default value
        if (modelSelect.options.length > 0) {
            const defaultModel = config.default_model || config.models[0].name;
            modelSelect.value = defaultModel;
            currentModel = defaultModel;
        }
    }

    // Add event listener for model selection change
    document.getElementById('model-select').addEventListener('change', function() {
        if (this.value) {
            currentModel = this.value;
        }
    });
}); 