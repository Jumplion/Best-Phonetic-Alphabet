// static/script.js

const inputsDiv = document.getElementById('inputs');
const resultPre = document.getElementById('result');

// Create inputs for A-Z
for (let i = 65; i <= 90; i++) {
  const letter = String.fromCharCode(i);
  const input = document.createElement('input');
  input.placeholder = letter;
  input.id = letter;
  inputsDiv.appendChild(input);
  inputsDiv.appendChild(document.createElement('br'));
}

// Submit form
document.getElementById('alphabet-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const alphabet = {};
  for (let i = 65; i <= 90; i++) {
    const letter = String.fromCharCode(i);
    alphabet[letter] = document.getElementById(letter).vavlue;
  }

  const res = await fetch('/grade', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(alphabet)
  });
  const data = await res.json();
  resultPre.innerText = JSON.stringify(data, null, 2);
});