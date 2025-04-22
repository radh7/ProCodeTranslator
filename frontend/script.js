document.addEventListener("DOMContentLoaded", function() {
    const line1 = document.getElementById("line1");
    const line2 = document.getElementById("line2");
    const button = document.getElementById("get-started-btn");
  
    const text1 = "Welcome to";
    const text2 = "ProCode Translator";
  
    let charIndex1 = 0;
    let charIndex2 = 0;
    const typingSpeed = 100;  // Delay between each letter (in ms)
  
    function typeLine1() {
      if (charIndex1 < text1.length) {
        // Use innerHTML to handle spaces as non-breaking spaces
        line1.innerHTML += (text1.charAt(charIndex1) === ' ') ? '&nbsp;' : text1.charAt(charIndex1);
        charIndex1++;
        setTimeout(typeLine1, typingSpeed);
      } else {
        setTimeout(typeLine2, typingSpeed * 2);  // Start typing the second line after a delay
      }
    }
  
    function typeLine2() {
        if (charIndex2 < text2.length) {
          // Use innerHTML to handle spaces as non-breaking spaces
          line2.innerHTML += (text2.charAt(charIndex2) === ' ') ? '&nbsp;' : text2.charAt(charIndex2);
          charIndex2++;
          setTimeout(typeLine2, typingSpeed);
        } else {
          setTimeout(showButton, typingSpeed * 2);  // Show button after typing is done
        }
      }
    
      function showButton() {
        button.style.display = 'inline-block';  // Make the "Get Started" button visible
        setTimeout(() => {
          button.classList.add('show');  // Trigger the fade-in effect
        }, 100);  // Delay the fade-in slightly
      }
    
      typeLine1();  // Start typing the first line when the page loads
    });
  