/**
 * Enhanced Chat Interface JavaScript
 * Provides interactive functionality for the AI chat interface
 */

document.addEventListener('DOMContentLoaded', function() {
    // UI Elements
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const messagesContainer = document.getElementById('messages-wrap'); // Container chính để cuộn
    
    // Tạo phần tử messages-content nếu chưa có
    let messagesContent = messagesContainer.querySelector('.messages-container');
    if (!messagesContent) {
        messagesContent = document.createElement('div');
        messagesContent.className = 'messages-container';
        messagesContainer.appendChild(messagesContent);
    }
    
    const providerButtons = document.querySelectorAll('.provider-btn');
    const newChatButton = document.getElementById('new-chat-btn');
    const clearChatButton = document.getElementById('clear-chat-btn');
    const clearHistoryButton = document.getElementById('clear-history-btn');
    const chatThreads = document.querySelector('.chat-history');
    
    // Theme switcher
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    const htmlElement = document.documentElement;
    const body = document.body;
    
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
        
        // Hiển thị thông báo chào mừng ban đầu nếu không có thread nào
        initWelcomeMessage();
        
        // Ensure chat scrolls to bottom on initial load
        scrollToBottom(200);
        
        // Thêm event listener cho window load để đảm bảo cuộn xuống sau khi trang đã tải hoàn toàn
        window.addEventListener('load', () => {
            scrollToBottom(300);
        });
        
        // Thêm MutationObserver để theo dõi thay đổi trong container tin nhắn
        if ("MutationObserver" in window) {
            const observer = new MutationObserver((mutations) => {
                let shouldScroll = false;
                
                mutations.forEach(mutation => {
                    if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                        shouldScroll = true;
                    }
                });
                
                if (shouldScroll) {
                    scrollToBottom(10);
                    setTimeout(() => messagesContainer.scrollTop = messagesContainer.scrollHeight, 200);
                }
            });
            
            observer.observe(messagesContainer, { 
                childList: true, 
                subtree: true 
            });
        }
    }
    
    async function loadModels() {
        try {
            console.log('Fetching models from /chat/models...');
            const response = await fetch('/chat/models');
            
            if (!response.ok) {
                console.error(`API error: ${response.status} ${response.statusText}`);
                throw new Error(`Failed to load models: ${response.status}`);
            }

            const data = await response.json();
            console.log('Models loaded successfully:', data);
            
            // Transform the data into our expected format
            modelConfigs = {
                vimrc: {
                    models: data.vimrc.models.map(model => {
                        // Nếu model là object với name và display_name
                        if (typeof model === 'object' && model.name && model.display_name) {
                            return model;
                        }
                        // Nếu model chỉ là string
                        return {
                            name: model,
                            display_name: formatDisplayName(model, 'vimrc')
                        };
                    }),
                    default_model: data.vimrc.default
                },
                openai: {
                    models: data.openai.models.map(model => {
                        if (typeof model === 'object' && model.name && model.display_name) {
                            return model;
                        }
                        return {
                            name: model,
                            display_name: formatDisplayName(model, 'openai')
                        };
                    }),
                    default_model: data.openai.default
                },
                gemini: {
                    models: data.gemini.models.map(model => {
                        if (typeof model === 'object' && model.name && model.display_name) {
                            return model;
                        }
                        return {
                            name: model,
                            display_name: formatDisplayName(model, 'gemini')
                        };
                    }),
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
                        { name: 'vi-mrc-large', display_name: 'HTC FinBot' }
                    ],
                    default_model: 'vi-mrc-large'
                },
                openai: {
                    models: [
                        { name: 'gpt-3.5-turbo', display_name: 'GPT-3.5 Turbo' },
                        { name: 'gpt-4', display_name: 'GPT-4' }
                    ],
                    default_model: 'gpt-3.5-turbo'
                },
                gemini: {
                    models: [
                        { name: 'gemini-pro', display_name: 'Gemini Pro' }
                    ],
                    default_model: 'gemini-pro'
                }
            };
            updateModelOptions(currentProvider);
        }
    }
    
    async function updateProviderInfo() {
        const botNameElement = document.querySelector('.chat-title');
        if (botNameElement) {
            botNameElement.textContent = getProviderDisplayName();
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
        if (!messagesContent) {
            console.error('Cannot load thread: container not found');
            return;
        }
        
        if (threads[threadId]) {
            messages = threads[threadId].messages;
            currentThreadId = threadId;
            
            // Update UI
            renderMessages();
            
            // Update active thread in sidebar
            document.querySelectorAll('.thread-item').forEach(item => {
                item.classList.remove('active');
            });
            
            const threadElement = document.querySelector(`.thread-item[data-thread-id="${threadId}"]`);
            if (threadElement) {
                threadElement.classList.add('active');
            }
            
            // Cuộn xuống cuối
            scrollToBottom(100);
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
        if (!messagesContent) {
            console.error('Cannot create new thread: container not found');
            return;
        }
        
        // Clear current messages
        messages = [];
        currentThreadId = generateThreadId();
        
        // Clear message container
        messagesContent.innerHTML = '';
        
        // Create welcome message
        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'welcome-message';
        welcomeDiv.innerHTML = `
            <div class="message-content">
                <h3>Chào mừng đến với HTC FinBot</h3>
                <p>Hãy đặt câu hỏi để bắt đầu trò chuyện!</p>
            </div>
        `;
        messagesContent.appendChild(welcomeDiv);
        
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
        if (messageInput) {
            messageInput.value = '';
            messageInput.style.height = 'auto';
            if (sendButton) sendButton.disabled = true;
        }
    }
    
    function clearCurrentChat() {
        if (!confirm('Are you sure you want to clear the current chat?')) return;
        
        if (!messagesContent) {
            console.error('Cannot clear chat: container not found');
            return;
        }

        // Clear only current thread messages
        messages = [];
        messagesContent.innerHTML = '';
        
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
        messagesContent.appendChild(emptyDiv);
        
        // Clear input
        if (messageInput) {
            messageInput.value = '';
            messageInput.style.height = 'auto';
            if (sendButton) sendButton.disabled = true;
        }
    }

    function clearAllHistory() {
        if (!confirm('Are you sure you want to clear all chat history? This cannot be undone.')) return;

        // Clear all threads
        threads = {};
        localStorage.removeItem('chatThreads');
        
        // Clear current messages
        messages = [];
        messagesContent.innerHTML = '';
        
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
                threadElement.querySelector('.thread-content').textContent = threadTitle;
            }
            
            saveThreads();
        }
        
        // Show loading indicator
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message assistant-message loading';
        loadingDiv.innerHTML = `
            <div class="message-header">
                <span class="message-sender">${getProviderDisplayName()}</span>
                <span class="message-time">Đang trả lời...</span>
            </div>
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>`;
        messagesContent.appendChild(loadingDiv);
        scrollToBottom(10);
        
        try {
            const apiMessages = messages.map(msg => ({
                role: msg.role,
                content: msg.content,
                context: msg.context || null
            }));
            
            const response = await fetch('/chat/send', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    messages: apiMessages,
                    provider: currentProvider,
                    model: currentModel,
                    temperature: 0.7,
                    max_tokens: 800,
                    use_training_data: currentProvider === 'vimrc' // Chỉ sử dụng dữ liệu training khi provider là vimrc
                })
            });
            
            messagesContent.removeChild(loadingDiv);
            
            if (response.ok) {
                const data = await response.json();
                addMessage('assistant', data.content);
                Prism.highlightAll();
                // Scroll to bottom after AI response is displayed
                scrollToBottom(100);
            } else {
                const errorData = await response.json();
                addMessage('assistant', `Error: ${errorData.detail || 'Failed to get response'}`);
                scrollToBottom(50);
            }
        } catch (error) {
            if (messagesContent.contains(loadingDiv)) {
                messagesContent.removeChild(loadingDiv);
            }
            console.error('Error sending message:', error);
            addMessage('assistant', `Error: ${error.message || 'Something went wrong'}`);
            scrollToBottom(50);
        }
    }
    
    function renderMessages() {
        // Kiểm tra messagesContent tồn tại
        if (!messagesContent) {
            console.error('Messages content container not found!');
            return;
        }
        
        // Clear messages container
        messagesContent.innerHTML = '';
        
        // If no messages, show welcome message
        if (messages.length === 0) {
            const welcomeDiv = document.createElement('div');
            welcomeDiv.className = 'welcome-message';
            welcomeDiv.innerHTML = `
                <h3>Chào mừng đến với HTC FinBot</h3>
                <p>Hãy chọn nhà cung cấp AI và đặt câu hỏi để bắt đầu trò chuyện!</p>
            `;
            messagesContent.appendChild(welcomeDiv);
            scrollToBottom(50);
            return;
        }
        
        // Render all messages
        messages.forEach(message => {
            renderMessage(message);
        });
        
        // Apply syntax highlighting
        Prism.highlightAll();
        
        // Cuộn xuống dưới
        scrollToBottom(100);
    }
    
    function renderMessage(message) {
        if (!messagesContent) {
            console.error('Cannot render message: container not found');
            return;
        }
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${message.role}-message`;
        
        let displayName = message.role === 'user' ? 'You' : getProviderDisplayName();
        let timestamp = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        
        // Format message content - detect and format code blocks
        let formattedContent = formatMessageContent(message.content);
        
        messageDiv.innerHTML = `
            <div class="message-header">
                <span class="message-sender">${displayName}</span>
                <span class="message-time">${timestamp}</span>
            </div>
            <div class="message-content">${formattedContent}</div>
        `;
        
        messagesContent.appendChild(messageDiv);
        
        // Apply syntax highlighting
        Prism.highlightAllUnder(messageDiv);
    }
    
    function formatMessageContent(content) {
        // Replace code blocks with properly formatted HTML
        let formattedContent = content;
        
        // Match Markdown style code blocks with language
        formattedContent = formattedContent.replace(/```([\w-]+)?\n([\s\S]*?)\n```/g, (match, language, code) => {
            const lang = language || 'plaintext';
            return `<pre><code class="language-${lang}">${escapeHtml(code)}</code></pre>`;
        });
        
        // Match inline code
        formattedContent = formattedContent.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Format links
        formattedContent = formattedContent.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
        
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

    // Format display name for model selection
    function formatDisplayName(name, provider) {
        // Special case for HTC FinBot models
        if (provider === 'vimrc') {
            return 'HTC FinBot';
        }
        
        // Regular formatting for other models
        return name
            .split(/[-_]/)
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    // Helper function to scroll to bottom of messages container with a more aggressive approach
    function scrollToBottom(delay = 0) {
        setTimeout(() => {
            if (!messagesContainer || !messagesContent) {
                console.error('Cannot scroll: container not found');
                return;
            }
            
            // Direct DOM access for scrolling
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            
            // Also use scrollIntoView on last child
            const lastMessage = messagesContent.lastElementChild;
            if (lastMessage) {
                try {
                    lastMessage.scrollIntoView({ behavior: 'smooth', block: 'end' });
                } catch (e) {
                    console.log('ScrollIntoView failed, using fallback');
                }
            }
        }, delay);
    }

    // Initialize the welcome message
    function initWelcomeMessage() {
        if (!messagesContent) {
            console.error('Cannot show welcome message: container not found');
            return;
        }
        
        if (messagesContent.childElementCount === 0) {
            const welcomeDiv = document.createElement('div');
            welcomeDiv.className = 'welcome-message';
            welcomeDiv.innerHTML = `
                <h3>Chào mừng đến với HTC FinBot</h3>
                <p>Hãy chọn nhà cung cấp AI và đặt câu hỏi để bắt đầu trò chuyện!</p>
            `;
            messagesContent.appendChild(welcomeDiv);
            scrollToBottom(50);
        }
    }
    
    // Get provider display name based on current selection
    function getProviderDisplayName() {
        switch(currentProvider) {
            case 'vimrc':
                return 'HTC FinBot Assistant';
            case 'openai':
                return 'OpenAI Assistant';
            case 'gemini':
                return 'Gemini Assistant';
            default:
                return 'AI Assistant';
        }
    }

    // Add global auto-scroll to bottom of messages container
    setInterval(() => {
        // Only auto-scroll if near bottom already (user isn't scrolling up to read)
        if (messagesContainer) {
            const isNearBottom = messagesContainer.scrollHeight - messagesContainer.scrollTop - messagesContainer.clientHeight < 150;
            if (isNearBottom) {
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
        }
    }, 1000);

    function addMessage(role, content, context = null) {
        if (!messagesContent) {
            console.error('Cannot add message: container not found');
            return;
        }
        
        // Add to messages array
        messages.push({ role, content, context });
        
        // Save to localStorage
        saveThreads();
        
        // Update UI
        renderMessage({ role, content });
        
        // Cuộn xuống cuối
        scrollToBottom(100);
    }

    // Function to update model options based on selected provider
    async function updateModelOptions(provider) {
        // Update the model selection dropdown based on provider
        const modelSelect = document.getElementById('model-select');
        if (!modelSelect) {
            console.error('Model select element not found');
            return;
        }
        
        // Clear existing options
        modelSelect.innerHTML = '';
        
        // Get provider config
        const config = modelConfigs[provider];
        if (!config || !Array.isArray(config.models)) {
            console.error('Invalid model configuration for provider', provider);
            return;
        }
        
        // Create and add new options
        config.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.name;
            option.textContent = model.display_name;
            option.setAttribute('data-provider', provider); // Store provider info in the option
            modelSelect.appendChild(option);
        });
        
        // Get providers for other models too
        if (provider === 'vimrc') {
            // Add OpenAI models
            if (modelConfigs.openai && modelConfigs.openai.models) {
                const optgroup = document.createElement('optgroup');
                optgroup.label = 'OpenAI Models';
                
                modelConfigs.openai.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.name;
                    option.textContent = model.display_name;
                    option.setAttribute('data-provider', 'openai'); // Store provider info
                    optgroup.appendChild(option);
                });
                
                modelSelect.appendChild(optgroup);
            }
            
            // Add Gemini models
            if (modelConfigs.gemini && modelConfigs.gemini.models) {
                const optgroup = document.createElement('optgroup');
                optgroup.label = 'Gemini Models';
                
                modelConfigs.gemini.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.name;
                    option.textContent = model.display_name;
                    option.setAttribute('data-provider', 'gemini'); // Store provider info
                    optgroup.appendChild(option);
                });
                
                modelSelect.appendChild(optgroup);
            }
        }
        
        // Set default model
        if (config.default_model) {
            modelSelect.value = config.default_model;
            currentModel = config.default_model;
        } else if (modelSelect.options.length > 0) {
            currentModel = modelSelect.options[0].value;
        }
        
        // Add change event listener to update the current model and provider
        modelSelect.addEventListener('change', function() {
            currentModel = this.value;
            
            // Check if we need to change provider based on selected model
            const selectedOption = this.options[this.selectedIndex];
            const modelProvider = selectedOption.getAttribute('data-provider');
            
            if (modelProvider && modelProvider !== currentProvider) {
                // Update provider if different
                currentProvider = modelProvider;
                
                // Update UI to reflect provider change
                providerButtons.forEach(btn => {
                    btn.classList.remove('active');
                    if (btn.getAttribute('data-provider') === currentProvider) {
                        btn.classList.add('active');
                    }
                });
                
                // Update bot name display
                const botNameElement = document.querySelector('.chat-title');
                if (botNameElement) {
                    botNameElement.textContent = getProviderDisplayName();
                }
            }
        });
    }

    // Initialize theme
    initTheme();
    
    // Function to initialize theme based on user preference or saved setting
    function initTheme() {
        const savedTheme = localStorage.getItem('theme');
        const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        
        // Apply dark theme if saved or preferred
        if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
            enableDarkMode();
        } else {
            disableDarkMode();
        }
    }
    
    // Enable dark mode
    function enableDarkMode() {
        body.classList.add('dark-theme');
        if (darkModeToggle) {
            darkModeToggle.innerHTML = '<i class="fas fa-sun"></i><span>Light Mode</span>';
        }
        localStorage.setItem('theme', 'dark');
    }
    
    // Disable dark mode
    function disableDarkMode() {
        body.classList.remove('dark-theme');
        if (darkModeToggle) {
            darkModeToggle.innerHTML = '<i class="fas fa-moon"></i><span>Dark Mode</span>';
        }
        localStorage.setItem('theme', 'light');
    }
    
    // Toggle dark mode
    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', function() {
            if (body.classList.contains('dark-theme')) {
                disableDarkMode();
            } else {
                enableDarkMode();
            }
        });
    }
    
    // Toggle sidebar on mobile
    const toggleSidebarBtn = document.getElementById('toggle-sidebar');
    const sidebar = document.querySelector('.sidebar');
    
    if (toggleSidebarBtn && sidebar) {
        toggleSidebarBtn.addEventListener('click', function() {
            sidebar.classList.toggle('show');
        });
        
        // Close sidebar when clicking outside
        document.addEventListener('click', function(event) {
            if (sidebar.classList.contains('show') && 
                !sidebar.contains(event.target) && 
                event.target !== toggleSidebarBtn) {
                sidebar.classList.remove('show');
            }
        });
    }
}); 