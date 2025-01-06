import './style.css'
import javascriptLogo from './javascript.svg'
import viteLogo from '/vite.svg'
import { setupCounter } from './counter.js'

document.getElementById('reviewForm').addEventListener('submit', async function(event) {
  event.preventDefault();
  const review = document.getElementById('review').value;
  const response = await fetch('http://localhost:5000:5173', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ review: review })
  });
  const data = await response.json();
  document.getElementById('result').innerText = `Sentiment: ${data.sentiment}, Probability: ${data.probability}`;
});
