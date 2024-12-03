// app.js

// Function to analyze the input sentence
function analyzeSentence() {
    const sentenceInput = document.getElementById('sentenceInput');
    const sentence = sentenceInput.value.trim();

    if (!sentence) {
        showToast("Please enter some text before analyzing.");
        return;
    }

    const uploadMessage = document.getElementById('uploadMessage');
    uploadMessage.classList.remove('d-none');

    // Show loading spinner
    document.getElementById('loadingSpinner').classList.remove('d-none');

    fetch('/checkSentiment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sentence })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'An error occurred');
            });
        }
        return response.json();
    })
    .then(data => {
        const requestId = data.request_id;
        // Keep checking until the analysis is available
        fetchAnalysisWithRetry(requestId, 10);  // Max retries: 10
    })
    .catch(error => {
        console.error('Error:', error);
        uploadMessage.classList.add('d-none'); // Hide upload message on error
        document.getElementById('loadingSpinner').classList.add('d-none'); // Hide loading spinner
        showToast(error.message);
    });
}

let recognition;
let recognizing = false;

function startDictation() {
    const speakButton = document.getElementById("speakButton");

    if (!('webkitSpeechRecognition' in window)) {
        showToast("Your browser does not support speech recognition. Please try Chrome.");
        return;
    }

    if (!recognition) {
        // Initialize speech recognition
        recognition = new webkitSpeechRecognition();
        recognition.continuous = false; // Stop when user stops speaking
        recognition.interimResults = false;
        recognition.lang = "en-US";

        recognition.onstart = function () {
            recognizing = true;
            speakButton.innerText = "Listening...";
            speakButton.disabled = true; // Disable the button while recording
            console.log("Voice recognition started.");
        };

        recognition.onresult = function (event) {
            let transcript = event.results[0][0].transcript;
            document.getElementById("sentenceInput").value += transcript + ' ';
        };

        recognition.onerror = function (event) {
            console.error("Recognition error:", event.error);
            recognizing = false;
            speakButton.innerText = "🎤 Speak";
            speakButton.disabled = false;
        };

        recognition.onend = function () {
            recognizing = false;
            speakButton.innerText = "🎤 Speak";
            speakButton.disabled = false; // Re-enable the button
            console.log("Voice recognition ended.");
        };
    }

    if (!recognizing) {
        recognition.start();
    }
}

// Function to fetch analysis with retries
function fetchAnalysisWithRetry(requestId, retries) {
    if (retries <= 0) {
        console.error("Failed to retrieve analysis after multiple attempts");
        showToast("Analysis failed or is taking too long. Please try again later.");
        document.getElementById('uploadMessage').classList.add('d-none');
        document.getElementById('loadingSpinner').classList.add('d-none');
        return;
    }

    setTimeout(() => {
        fetch(`/get_analysis?request_id=${requestId}`)
            .then(response => {
                if (!response.ok) {
                    return response.json().then(data => {
                        throw new Error(data.error || data.message || 'An error occurred');
                    });
                }
                return response.json();
            })
            .then(data => {
                if (data.message) {
                    console.error(data.message);
                    // Retry if no analysis found yet
                    fetchAnalysisWithRetry(requestId, retries - 1);
                } else {
                    // Pass the data to visualization functions
                    visualizeResults(data);
                    document.getElementById('uploadMessage').classList.add('d-none'); // Hide upload message
                    document.getElementById('loadingSpinner').classList.add('d-none'); // Hide loading spinner
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showToast(error.message);
                document.getElementById('uploadMessage').classList.add('d-none');
                document.getElementById('loadingSpinner').classList.add('d-none');
            });
    }, 5000);  // Retry interval set to 5 seconds
}

// Function to visualize all results
function visualizeResults(data) {
    // Hide the upload message and loading spinner
    document.getElementById('uploadMessage').classList.add('d-none');
    document.getElementById('loadingSpinner').classList.add('d-none');

    // Show the results section
    document.querySelector('.results-section').classList.remove('d-none');

    // Visualization functions
    displaySummary(data);
    visualizeTopics(data);
    visualizeOverallEmotions(data);
    visualizeSentimentTrend(data);
    visualizeSentimentIntensity(data);
    visualizeSentimentDistribution(data);
    visualizeEmotionalShifts(data);
    visualizeEntities(data);
}

// Function to display the summary text
function displaySummary(data) {
    document.getElementById('summaryText').textContent = data.summary;
}


// Function to visualize topics using a word cloud
function visualizeTopics(data) {
    const topics = data.topics;
    // Flatten the topics into words and weights
    let words = [];
    topics.forEach(topic => {
        // topic[1] is the topic string
        const wordWeights = topic[1].split('+').map(item => {
            const parts = item.trim().split('*');
            return {
                text: parts[1].replace(/"/g, ''),
                size: parseFloat(parts[0]) * 1000  // Scale the size appropriately
            };
        });
        words = words.concat(wordWeights);
    });

    // Get container dimensions
    const container = document.getElementById('topics');
    const width = container.offsetWidth;
    const height = 300;

    // Remove any existing SVG
    d3.select("#topics").selectAll("*").remove();

    const svg = d3.select("#topics").append("svg")
        .attr("width", width)
        .attr("height", height);

    const layout = d3.layout.cloud()
        .size([width, height])
        .words(words)
        .padding(5)
        .rotate(() => ~~(Math.random() * 2) * 90)
        .fontSize(d => d.size)
        .on("end", draw);

    layout.start();

    function draw(words) {
        svg.append("g")
            .attr("transform", `translate(${width / 2}, ${height / 2})`)
            .selectAll("text")
            .data(words)
            .enter().append("text")
            .style("font-size", d => `${d.size}px`)
            .style("fill", () => `hsl(${Math.random() * 360},100%,50%)`)
            .attr("text-anchor", "middle")
            .attr("transform", d => `translate(${[d.x, d.y]})rotate(${d.rotate})`)
            .text(d => d.text);
    }
}

// Function to visualize overall emotions using a pie chart
function visualizeOverallEmotions(data) {
    const sentences = data.sentences;

    // Classify each sentence based on the compound score
    const sentimentCounts = { Positive: 0, Negative: 0, Neutral: 0 };

    sentences.forEach(sentence => {
        const compoundScore = sentence.analysis.compound;
        if (compoundScore >= 0.05) {
            sentimentCounts.Positive += 1;
        } else if (compoundScore <= -0.05) {
            sentimentCounts.Negative += 1;
        } else {
            sentimentCounts.Neutral += 1;
        }
    });

    const totalSentences = sentences.length;
    const pieData = [
        { sentiment: 'Positive', count: sentimentCounts.Positive },
        { sentiment: 'Negative', count: sentimentCounts.Negative },
        { sentiment: 'Neutral', count: sentimentCounts.Neutral }
    ];

    // Get container dimensions
    const container = document.getElementById('overallEmotions');
    const width = container.offsetWidth;
    const height = 300;
    const radius = Math.min(width, height) / 2 - 20;

    // Remove existing SVG
    d3.select("#overallEmotions").selectAll("*").remove();

    const svg = d3.select("#overallEmotions").append("svg")
        .attr("width", width)
        .attr("height", height);

    const chartArea = svg.append("g")
        .attr("transform", `translate(${width / 2 - 50}, ${height / 2})`); // Adjusted translation

    const color = d3.scaleOrdinal()
        .domain(pieData.map(d => d.sentiment))
        .range(['#198754', '#dc3545', '#6c757d']); // Bootstrap colors

    const pie = d3.pie()
        .value(d => d.count);

    const data_ready = pie(pieData);

    // Build the pie chart
    chartArea.selectAll('whatever')
        .data(data_ready)
        .enter()
        .append('path')
        .attr('d', d3.arc()
            .innerRadius(0)
            .outerRadius(radius)
        )
        .attr('fill', d => color(d.data.sentiment))
        .attr("stroke", "#fff")
        .style("stroke-width", "2px");

    // Add percentage labels
    chartArea.selectAll('mySlices')
        .data(data_ready)
        .enter()
        .append('text')
        .text(d => {
            const percent = ((d.data.count / totalSentences) * 100).toFixed(1);
            return `${percent}%`;
        })
        .attr("transform", d => `translate(${d3.arc()
            .innerRadius(0)
            .outerRadius(radius * 0.6)
            .centroid(d)})`)
        .style("text-anchor", "middle")
        .style("font-size", 14)
        .style("fill", "#fff");

    // Add legend inside the SVG but within the container width
    const legend = svg.append("g")
        .attr("transform", `translate(${width / 2 + radius - 30}, ${height / 2 - radius})`); // Adjusted position

    legend.selectAll(".legend-item")
        .data(pieData)
        .enter()
        .append("g")
        .attr("class", "legend-item")
        .attr("transform", (d, i) => `translate(0, ${i * 25})`)
        .call(g => {
            g.append("rect")
                .attr("width", 18)
                .attr("height", 18)
                .attr("fill", d => color(d.sentiment));

            g.append("text")
                .attr("x", 24)
                .attr("y", 14)
                .text(d => `${d.sentiment} (${d.count})`)
                .style("font-size", 14);
        });
}

// Function to visualize sentiment trend using a line chart
function visualizeSentimentTrend(data) {
    const sentimentTrend = data.sentiment_trend;

    // Remove previous visualization
    d3.select("#sentimentTrend").selectAll("*").remove();

    // Get container dimensions
    const container = document.getElementById('sentimentTrend');
    const width = container.offsetWidth;
    const height = 300;
    const margin = { top: 20, right: 20, bottom: 50, left: 50 };

    const svg = d3.select("#sentimentTrend")
        .append("svg")
        .attr("width", width)
        .attr("height", height + margin.top + margin.bottom);

    const chartAreaWidth = width - margin.left - margin.right - 100; // Space for legend
    const chartArea = svg.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Create scales
    const x = d3.scaleLinear()
        .domain(d3.extent(sentimentTrend, d => d.sentence_index))
        .range([0, chartAreaWidth]);

    const y = d3.scaleLinear()
        .domain([-1, 1])
        .range([height, 0]);

    // Add X axis
    chartArea.append("g")
        .attr("transform", `translate(0,${height})`)
        .call(d3.axisBottom(x).ticks(10))
        .append("text")
        .attr("x", chartAreaWidth / 2)
        .attr("y", 40)
        .attr("fill", "#000")
        .style("font-size", "14px")
        .style("text-anchor", "middle")
        .text("Sentence Number");

    // Add Y axis
    chartArea.append("g")
        .call(d3.axisLeft(y))
        .append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -height / 2)
        .attr("y", -40)
        .attr("fill", "#000")
        .style("font-size", "14px")
        .style("text-anchor", "middle")
        .text("Compound Sentiment Score");

    // Add line path
    chartArea.append("path")
        .datum(sentimentTrend)
        .attr("fill", "none")
        .attr("stroke", "#0d6efd") // Bootstrap primary color
        .attr("stroke-width", 2)
        .attr("d", d3.line()
            .x(d => x(d.sentence_index))
            .y(d => y(d.compound))
        );

    // Add points
    chartArea.selectAll("dot")
        .data(sentimentTrend)
        .enter()
        .append("circle")
        .attr("cx", d => x(d.sentence_index))
        .attr("cy", d => y(d.compound))
        .attr("r", 4)
        .attr("fill", "#0d6efd");

    // Add legend inside the SVG but within the container width
    const legend = chartArea.append("g")
        .attr("transform", `translate(${chartAreaWidth + 20}, ${20})`);

    legend.append("rect")
        .attr("width", 18)
        .attr("height", 18)
        .attr("fill", "#0d6efd");

    legend.append("text")
        .attr("x", 24)
        .attr("y", 14)
        .text("Compound Sentiment Score")
        .style("font-size", 14)
        .attr("dy", ".35em");
}

// Function to visualize sentiment intensity per sentence using a bar chart
function visualizeSentimentIntensity(data) {
    const sentences = data.sentences;
    const sentenceIndices = sentences.map((_, i) => i + 1); // Start from 1
    const compoundScores = sentences.map(sentence => sentence.analysis.compound);

    // Get container dimensions
    const container = document.getElementById('sentimentIntensity');
    const width = container.offsetWidth;
    const height = 300;
    const margin = { top: 20, right: 20, bottom: 70, left: 70 };

    // Remove existing SVG
    d3.select("#sentimentIntensity").selectAll("*").remove();

    const svg = d3.select("#sentimentIntensity")
        .append("svg")
        .attr("width", width)
        .attr("height", height + margin.top + margin.bottom);

    const chartAreaWidth = width - margin.left - margin.right - 100; // Space for legend
    const chartArea = svg.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Create scales
    const x = d3.scaleBand()
        .domain(sentenceIndices)
        .range([0, chartAreaWidth])
        .padding(0.1);

    const y = d3.scaleLinear()
        .domain([-1, 1])
        .range([height, 0]);

    // Add X axis
    chartArea.append("g")
        .attr("transform", `translate(0,${height})`)
        .call(d3.axisBottom(x).tickValues(x.domain().filter((d, i) => !(i % 5))))
        .append("text")
        .attr("x", chartAreaWidth / 2)
        .attr("y", 50)
        .attr("fill", "#000")
        .style("font-size", "14px")
        .style("text-anchor", "middle")
        .text("Sentence Number");

    // Add Y axis
    chartArea.append("g")
        .call(d3.axisLeft(y))
        .append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -height / 2)
        .attr("y", -50)
        .attr("fill", "#000")
        .style("font-size", "14px")
        .style("text-anchor", "middle")
        .text("Compound Sentiment Score");

    // Add bars
    chartArea.selectAll(".bar")
        .data(compoundScores)
        .enter()
        .append("rect")
        .attr("x", (d, i) => x(i + 1))
        .attr("width", x.bandwidth())
        .attr("y", d => y(Math.max(0, d)))
        .attr("height", d => Math.abs(y(d) - y(0)))
        .attr("fill", d => d >= 0 ? '#198754' : '#dc3545'); // Bootstrap success and danger colors

    // Add legend inside the SVG but within the container width
    const legendData = [
        { label: "Positive", color: "#198754" },
        { label: "Negative", color: "#dc3545" }
    ];

    const legend = chartArea.append("g")
        .attr("transform", `translate(${chartAreaWidth + 20}, ${20})`);

    legend.selectAll(".legend-item")
        .data(legendData)
        .enter()
        .append("g")
        .attr("class", "legend-item")
        .attr("transform", (d, i) => `translate(0, ${i * 25})`)
        .call(g => {
            g.append("rect")
                .attr("width", 18)
                .attr("height", 18)
                .attr("fill", d => d.color);

            g.append("text")
                .attr("x", 24)
                .attr("y", 14)
                .text(d => d.label)
                .style("font-size", 14)
                .attr("dy", ".35em");
        });
}

// Function to visualize sentiment distribution using a histogram
function visualizeSentimentDistribution(data) {
    const compoundScores = data.sentences.map(sentence => sentence.analysis.compound);

    // Get container dimensions
    const container = document.getElementById('sentimentDistribution');
    const width = container.offsetWidth;
    const height = 300;
    const margin = { top: 20, right: 20, bottom: 70, left: 70 };

    // Remove existing SVG
    d3.select("#sentimentDistribution").selectAll("*").remove();

    const svg = d3.select("#sentimentDistribution")
        .append("svg")
        .attr("width", width)
        .attr("height", height + margin.top + margin.bottom);

    const chartAreaWidth = width - margin.left - margin.right - 100; // Space for legend
    const chartArea = svg.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Create x-axis scale
    const x = d3.scaleLinear()
        .domain([-1, 1])  // Sentiment scores range
        .range([0, chartAreaWidth]);

    // Generate histogram data
    const histogram = d3.histogram()
        .domain(x.domain())
        .thresholds(x.ticks(20));

    const bins = histogram(compoundScores);

    // Create y-axis scale
    const y = d3.scaleLinear()
        .domain([0, d3.max(bins, d => d.length)])
        .range([height, 0]);

    // Add X axis
    chartArea.append("g")
        .attr("transform", `translate(0,${height})`)
        .call(d3.axisBottom(x))
        .append("text")
        .attr("x", chartAreaWidth / 2)
        .attr("y", 50)
        .attr("fill", "#000")
        .style("font-size", "14px")
        .style("text-anchor", "middle")
        .text("Compound Sentiment Score");

    // Add Y axis
    chartArea.append("g")
        .call(d3.axisLeft(y))
        .append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -height / 2)
        .attr("y", -50)
        .attr("fill", "#000")
        .style("font-size", "14px")
        .style("text-anchor", "middle")
        .text("Number of Sentences");

    // Add bars
    chartArea.selectAll("rect")
        .data(bins)
        .enter()
        .append("rect")
        .attr("x", d => x(d.x0) + 1)
        .attr("y", d => y(d.length))
        .attr("width", d => Math.max(0, x(d.x1) - x(d.x0) - 2))
        .attr("height", d => height - y(d.length))
        .style("fill", "#0d6efd"); // Bootstrap primary color

    // Add legend inside the SVG but within the container width
    const legend = chartArea.append("g")
        .attr("transform", `translate(${chartAreaWidth + 20}, ${20})`);

    legend.append("rect")
        .attr("width", 18)
        .attr("height", 18)
        .attr("fill", "#0d6efd");

    legend.append("text")
        .attr("x", 24)
        .attr("y", 14)
        .text("Sentiment Distribution")
        .style("font-size", 14)
        .attr("dy", ".35em");
}

// Function to visualize emotional shifts using a stacked area chart
function visualizeEmotionalShifts(data) {
    const sentences = data.sentences;
    const emotionTypes = ["Happy", "Angry", "Surprise", "Sad", "Fear", "Neutral"];
    const emotionData = [];

    sentences.forEach((sentence, index) => {
        const emotions = sentence.emotions;
        const emotionCounts = { sentence_index: index + 1 };
        emotionTypes.forEach(emotion => {
            emotionCounts[emotion] = emotions.includes(emotion) ? 1 : 0;
        });
        emotionData.push(emotionCounts);
    });

    // Prepare data for stack
    const stack = d3.stack().keys(emotionTypes);
    const stackedData = stack(emotionData);

    // Get container dimensions
    const container = document.getElementById('emotionalShifts');
    const width = container.offsetWidth;
    const height = 300;
    const margin = { top: 20, right: 20, bottom: 70, left: 70 };

    // Remove existing SVG
    d3.select("#emotionalShifts").selectAll("*").remove();

    const svg = d3.select("#emotionalShifts")
        .append("svg")
        .attr("width", width)
        .attr("height", height + margin.top + margin.bottom);

    const chartAreaWidth = width - margin.left - margin.right - 100; // Space for legend
    const chartArea = svg.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Create scales
    const x = d3.scaleLinear()
        .domain([1, sentences.length])
        .range([0, chartAreaWidth]);

    const y = d3.scaleLinear()
        .domain([0, d3.max(stackedData[stackedData.length - 1], d => d[1])]) // Adjusted to actual max value
        .range([height, 0]);

    // Create color scale
    const color = d3.scaleOrdinal()
        .domain(emotionTypes)
        .range(d3.schemeCategory10);

    // Add X axis
    chartArea.append("g")
        .attr("transform", `translate(0,${height})`)
        .call(d3.axisBottom(x).ticks(10))
        .append("text")
        .attr("x", chartAreaWidth / 2)
        .attr("y", 50)
        .attr("fill", "#000")
        .style("font-size", "14px")
        .style("text-anchor", "middle")
        .text("Sentence Number");

    // Add Y axis
    chartArea.append("g")
        .call(d3.axisLeft(y))
        .append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -height / 2)
        .attr("y", -50)
        .attr("fill", "#000")
        .style("font-size", "14px")
        .style("text-anchor", "middle")
        .text("Emotion Intensity");

    // Add area layers
    chartArea.selectAll(".layer")
        .data(stackedData)
        .enter()
        .append("path")
        .attr("class", "layer")
        .style("fill", d => color(d.key))
        .attr("d", d3.area()
            .x(d => x(d.data.sentence_index))
            .y0(d => y(d[0]))
            .y1(d => y(d[1]))
        );

    // Add legend inside the SVG but within the container width
    const legend = chartArea.append("g")
        .attr("transform", `translate(${chartAreaWidth + 20}, ${20})`);

    legend.selectAll(".legend-item")
        .data(emotionTypes)
        .enter()
        .append("g")
        .attr("class", "legend-item")
        .attr("transform", (d, i) => `translate(0, ${i * 25})`)
        .call(g => {
            g.append("rect")
                .attr("width", 18)
                .attr("height", 18)
                .attr("fill", d => color(d));

            g.append("text")
                .attr("x", 24)
                .attr("y", 14)
                .text(d => d)
                .style("font-size", 14)
                .attr("dy", ".35em");
        });
}

// Function to visualize Named Entities in a table format
function visualizeEntities(data) {
    const sentences = data.sentences;

    // Remove existing content
    const entitiesContainer = document.getElementById('entities');
    entitiesContainer.innerHTML = '';

    // Create table
    const table = document.createElement('table');
    table.classList.add('entities-table', 'table', 'table-striped', 'table-bordered');

    // Create table header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');

    const sentenceHeader = document.createElement('th');
    sentenceHeader.innerText = 'Sentence';
    headerRow.appendChild(sentenceHeader);

    const entityHeader = document.createElement('th');
    entityHeader.innerText = 'Entity';
    headerRow.appendChild(entityHeader);

    const attributeHeader = document.createElement('th');
    attributeHeader.innerText = 'Attribute';
    headerRow.appendChild(attributeHeader);

    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Create table body
    const tbody = document.createElement('tbody');

    sentences.forEach((sentenceEntry, index) => {
        const sentenceText = sentenceEntry.sentence;
        const entities = sentenceEntry.entities;

        if (entities && entities.length > 0) {
            entities.forEach(entity => {
                const row = document.createElement('tr');

                const sentenceCell = document.createElement('td');
                sentenceCell.innerText = `Sentence ${index + 1}: ${sentenceText}`;
                row.appendChild(sentenceCell);

                const entityCell = document.createElement('td');
                entityCell.innerText = entity.text;
                row.appendChild(entityCell);

                const attributeCell = document.createElement('td');
                attributeCell.innerText = entity.label;
                row.appendChild(attributeCell);

                tbody.appendChild(row);
            });
        } else {
            // If no entities, still display the sentence with 'No entities found'
            const row = document.createElement('tr');

            const sentenceCell = document.createElement('td');
            sentenceCell.innerText = `Sentence ${index + 1}: ${sentenceText}`;
            row.appendChild(sentenceCell);

            const entityCell = document.createElement('td');
            entityCell.innerText = 'No entities found';
            row.appendChild(entityCell);

            const attributeCell = document.createElement('td');
            attributeCell.innerText = '-';
            row.appendChild(attributeCell);

            tbody.appendChild(row);
        }
    });

    table.appendChild(tbody);
    entitiesContainer.appendChild(table);
}

// Function to show toast notifications
function showToast(message) {
    document.getElementById('toastBody').innerText = message;
    const toastElement = document.getElementById('liveToast');
    const toast = new bootstrap.Toast(toastElement);
    toast.show();
}

// Function to redo the analysis
function redoAnalysis() {
    document.getElementById('sentenceInput').value = '';
    d3.selectAll("svg").remove();
    document.getElementById('summaryText').textContent = '';
    document.getElementById('uploadMessage').classList.add('d-none');

    // Clear NER visualization
    const entitiesContainer = document.getElementById('entities');
    entitiesContainer.innerHTML = '';

    // Hide the results section
    document.querySelector('.results-section').classList.add('d-none');

    // Reset speech recognition state
    if (recognition && recognizing) {
        recognition.stop();
        recognizing = false;
        const speakButton = document.getElementById("speakButton");
        speakButton.innerText = "🎤 Speak";
        speakButton.disabled = false;
    }
}