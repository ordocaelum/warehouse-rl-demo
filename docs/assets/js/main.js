// Tab switching functionality
const navLinks = document.querySelectorAll('.nav-link');
const tabContents = document.querySelectorAll('.tab-content');

navLinks.forEach(link => {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    const tabName = link.getAttribute('data-tab');
    
    // Remove active class from all
    navLinks.forEach(l => l.classList.remove('active'));
    tabContents.forEach(tab => tab.classList.remove('active'));
    
    // Add active class to clicked
    link.classList.add('active');
    document.getElementById(tabName).classList.add('active');
  });
});

// Copy to clipboard functionality
function copyCode(button) {
  const codeBlock = button.previousElementSibling;
  const text = codeBlock.textContent;
  
  navigator.clipboard.writeText(text).then(() => {
    const originalText = button.textContent;
    button.textContent = 'Copied!';
    setTimeout(() => {
      button.textContent = originalText;
    }, 2000);
  });
}

// Collapsible sections
function toggleCollapsible(header) {
  const body = header.nextElementSibling;
  body.classList.toggle('active');
  
  // Add chevron animation
  header.style.transform = body.classList.contains('active') ? 'rotate(180deg)' : 'rotate(0)';
}

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      target.scrollIntoView({ behavior: 'smooth' });
    }
  });
});

// Initialize first tab as active
if (navLinks.length > 0) {
  navLinks[0].classList.add('active');
}
if (tabContents.length > 0) {
  tabContents[0].classList.add('active');
}
