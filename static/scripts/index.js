const DEFAULT_SENDER_NAME = 'You';
const DEFAULT_RECIPIENT_NAME = 'ELECTRA transformer';

const chatContainerElement = document.getElementById('chat-container');
const contextTextarea = document.getElementById('context-textarea');
const messageTextarea = document.getElementById('message-textarea');

let savedContext = '';

function createMessageBodyElement(message, senderName) {
    const messageBodyElement = document.createElement('div');
    messageBodyElement.classList.add('card-body');

    const nameElement = document.createElement('h6');
    nameElement.classList.add('card-title');
    nameElement.innerText = senderName;

    const messageTextElement = document.createElement('p');
    messageTextElement.classList.add('card-text');
    messageTextElement.innerText = message;

    messageBodyElement.appendChild(nameElement);
    messageBodyElement.appendChild(messageTextElement);

    return messageBodyElement;
}

function addMessage(messageElement, message, senderName) {
    messageElement.appendChild(createMessageBodyElement(message, senderName));
    chatContainerElement.appendChild(messageElement);
}

function createResponseChatMessage(message, senderName) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('card', 'text-white', 'w-50', 'm-4', 'bg-secondary');

    addMessage(messageElement, message, senderName);
}

function createSentChatMessage(message, senderName) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('card', 'text-white', 'w-50', 'm-4', 'text-end', 'float-end', 'bg-primary');

    addMessage(messageElement, message, senderName);
}

function scrollChatToBottom() {
    chatContainerElement.scrollTo(0, chatContainerElement.scrollHeight);
}

function sendQuestion() {
    const context = savedContext;
    const question = messageTextarea.value;

    createSentChatMessage(question, DEFAULT_SENDER_NAME);
    scrollChatToBottom();

    const xmlHttpRequest = new XMLHttpRequest();
    xmlHttpRequest.addEventListener('load', function() {
        createResponseChatMessage(JSON.parse(this.responseText).answer, DEFAULT_RECIPIENT_NAME);
        scrollChatToBottom();
    });
    xmlHttpRequest.open('GET', `http://127.0.0.1:5000/answer?context=${context}&question=${question}`);
    xmlHttpRequest.send();
}

function saveContext() {
    savedContext = contextTextarea.value;
    createSentChatMessage(`Context: ${savedContext}`, DEFAULT_SENDER_NAME);
    createResponseChatMessage('Ok, got it', DEFAULT_RECIPIENT_NAME);
    scrollChatToBottom();
}
