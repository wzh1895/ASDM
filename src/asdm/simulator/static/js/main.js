// main.js

// Grab DOM elements
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('fileElem');
const resultsContent = document.getElementById('results-content');
const downloadLink = document.getElementById('download-link');
const downloadSection = document.getElementById('download-section');

// New references for charting
const chartSection = document.getElementById('chart-section');
const varSelect = document.getElementById('varSelect');
const plotDiv = document.getElementById('plot');

// We'll store the simulation records in a global variable
let globalRecords = null;
let globalTimeCol = null;

// Prevent default drag behaviours on the page
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, preventDefaults, false);
  document.body.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

// Highlight drop area when file is dragged over it
['dragenter', 'dragover'].forEach(eventName => {
  dropArea.addEventListener(eventName, highlight, false);
});
['dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, unhighlight, false);
});

function highlight(e) {
  dropArea.classList.add('highlight');
}
function unhighlight(e) {
  dropArea.classList.remove('highlight');
}

// Handle dropped files
dropArea.addEventListener('drop', handleDrop, false);
// Also handle click on the drop area -> triggers file open dialog
dropArea.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => handleFiles(fileInput.files));

function handleDrop(e) {
  let dt = e.dataTransfer;
  let files = dt.files;
  handleFiles(files);
}

function handleFiles(files) {
  const file = files[0];
  if (!file) return;

  // Prepare form data
  let formData = new FormData();
  formData.append('model_file', file);

  // Send to server
  fetch('/simulate', {
    method: 'POST',
    body: formData
  })
  .then(res => res.json())
  .then(data => {
    console.log("Server response:", data);
    
    // Handle errors
    if (data.error) {
      resultsContent.innerHTML = `<p style="color:red;">Error: ${data.error}</p>`;

      // Show full error logs (call stack)
      let errorLog = data.error_log ? `<pre>${data.error_log}</pre>` : "No additional error details.";
      document.getElementById('error-content').innerHTML = errorLog; // Use innerHTML to preserve formatting
      document.getElementById('error-section').style.display = 'block';
      document.getElementById('error-section').open = true;  // Unfold error section
      return;
    }

    // Store the results globally
    globalRecords = data.data || [];
    globalTimeCol = data.time_col || "Time"; 

    // 1) Show the results section
    if (globalRecords.length > 0) {
      document.getElementById('results').style.display = 'block';
      displayResults(globalRecords);
    }

    // 2) Show the download section if there's a CSV file
    if (data?.download_url) {
      downloadLink.href = data.download_url;
      downloadSection.style.display = 'block';
      downloadSection.open = true;
    }

    // 3) Show the visualization section
    if (globalRecords.length > 0) {
      setupChartOptions(globalRecords, globalTimeCol);
      chartSection.style.display = 'block';
      chartSection.open = true;
    }

    // Hide error logs if the previous run had errors
    document.getElementById('error-section').style.display = 'none';
  })
  .catch(err => {
    console.error(err);
    resultsContent.innerHTML = '<p style="color:red;">Error occurred.</p>';

    // Show full error message for client-side issues
    let errorLog = err.stack ? `<pre>${err.stack}</pre>` : "Unexpected error occurred.";
    document.getElementById('error-content').innerHTML = errorLog;
    document.getElementById('error-section').style.display = 'block';
    document.getElementById('error-section').open = true;  // Unfold error section
  });
}

function displayResults(records) {
  // If there's no 'data' or it's empty
  if (!records) {
    resultsContent.innerHTML = '<p>No data returned.</p>';
    return;
  }
  if (!records.length) {
    resultsContent.innerHTML = '<p>No data returned (empty array).</p>';
    return;
  }

  // Build a table
  let html = '<table><thead><tr>';
  const headers = Object.keys(records[0]);
  headers.forEach(h => {
    html += `<th>${h}</th>`;
  });
  html += '</tr></thead><tbody>';

  records.forEach(row => {
    html += '<tr>';
    headers.forEach(h => {
      html += `<td>${row[h]}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';

  resultsContent.innerHTML = html;
}

function setupChartOptions(records) {
  // Clear any existing <option> in varSelect
  varSelect.innerHTML = '';

  // We'll guess that the first object has all the keys
  // In a typical SD model, there's often a "Time" column
  // plus other variables
  const headers = Object.keys(records[0]);

  // We can exclude a "time" field from the dropdown if we like:
  // or specifically exclude other fields we don't want to plot
  headers.forEach(h => {
    // if (h.toLowerCase() === 'time') return; // example of skipping
    let option = document.createElement('option');
    option.value = h;
    option.textContent = h;
    varSelect.appendChild(option);
  });

  // Listen for changes
  varSelect.addEventListener('change', () => {
    plotVariable(records, varSelect.value);
  });

  // Default: plot the first variable (or skip if you want)
  if (varSelect.options.length > 0) {
    plotVariable(records, varSelect.options[0].value);
  }
}

function plotVariable(records, varName) {
  console.log(`Plotting variable: ${varName} against ${globalTimeCol}`);

  // Extract time column values
  const xValues = records.map(row => row[globalTimeCol] !== undefined ? row[globalTimeCol] : null);
  const yValues = records.map(row => row[varName]);

  let trace = {
    x: xValues,
    y: yValues,
    mode: 'lines',
    type: 'scatter',
    name: varName
  };

  let layout = {
    title: `Plot of ${varName} over ${globalTimeCol}`, // Updated title format
    xaxis: { title: globalTimeCol },
    yaxis: { title: varName }
  };

  Plotly.newPlot(plotDiv, [trace], layout);
}
