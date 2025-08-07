// Make the logo clickable
document.addEventListener('DOMContentLoaded', function() {
    const logo = document.querySelector('.sidebar-brand img');
    if (logo) {
        logo.style.cursor = 'pointer';
        logo.style.maxWidth = '120px';
        logo.style.maxHeight = '120px';
        logo.addEventListener('click', function() {
            window.location.href = 'https://github.com/Andrew-XQY/XFlow';
        });
    }
    
    // Force change document title
    document.title = 'Documentation';
    
    // Try to change any header titles
    const titleElements = document.querySelectorAll('title, .sidebar-brand-text, h1');
    titleElements.forEach(el => {
        if (el.textContent.includes('XFlow Documentation')) {
            el.textContent = 'Documentation';
        }
    });
});
