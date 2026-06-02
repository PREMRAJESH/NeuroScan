let selectedFile = null;
let modelReady = false;

const uploadArea = document.getElementById("uploadArea");
const browseButton = document.getElementById("browseButton");
const fileInput = document.getElementById("fileInput");
const imagePreview = document.getElementById("imagePreview");
const previewImg = document.getElementById("previewImg");
const fileName = document.getElementById("fileName");
const fileMeta = document.getElementById("fileMeta");
const removeImageBtn = document.getElementById("removeImage");
const analyzeButton = document.getElementById("analyzeButton");
const workspaceSection = document.getElementById("workspace");
const resultsSection = document.getElementById("resultsSection");
const resultContent = document.getElementById("resultContent");
const emptyResult = document.getElementById("emptyResult");
const emptyResultText = document.getElementById("emptyResultText");
const resultBadge = document.getElementById("resultBadge");
const errorMessage = document.getElementById("errorMessage");
const errorText = document.getElementById("errorText");
const newScanButton = document.getElementById("newScanButton");
const modelStatus = document.getElementById("modelStatus");
const modelStatusText = document.getElementById("modelStatusText");
const classGrid = document.getElementById("classGrid");
const artifactGrid = document.getElementById("artifactGrid");
const navToggle = document.getElementById("navToggle");
const mobileNav = document.getElementById("mobileNav");

document.addEventListener("DOMContentLoaded", () => {
    setWorkspaceStage("empty");
    setupNavigation();
    setupUploadFlow();
    loadModelInfo();
    setupLightbox();
});

function setupNavigation() {
    const navLinks = Array.from(document.querySelectorAll(".side-nav a, .mobile-nav a"));

    navToggle.addEventListener("click", () => {
        const isOpen = document.body.classList.toggle("nav-open");
        navToggle.setAttribute("aria-expanded", String(isOpen));
    });

    navLinks.forEach((link) => {
        link.addEventListener("click", () => {
            document.body.classList.remove("nav-open");
            navToggle.setAttribute("aria-expanded", "false");
        });
    });

    const sections = Array.from(document.querySelectorAll("main > section[id]"));

    if (!("IntersectionObserver" in window)) {
        return;
    }

    const observer = new IntersectionObserver(
        (entries) => {
            const visible = entries
                .filter((entry) => entry.isIntersecting)
                .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

            if (!visible) {
                return;
            }

            navLinks.forEach((link) => {
                link.classList.toggle("active", link.getAttribute("href") === `#${visible.target.id}`);
            });
        },
        { rootMargin: "-24% 0px -58% 0px", threshold: [0.1, 0.35, 0.65] }
    );

    sections.forEach((section) => observer.observe(section));
}

function setupUploadFlow() {
    fileInput.addEventListener("change", handleFileSelect);

    browseButton.addEventListener("click", (event) => {
        event.stopPropagation();
        fileInput.click();
    });

    uploadArea.addEventListener("click", () => fileInput.click());
    uploadArea.addEventListener("dragover", handleDragOver);
    uploadArea.addEventListener("dragleave", handleDragLeave);
    uploadArea.addEventListener("drop", handleDrop);

    removeImageBtn.addEventListener("click", resetUpload);
    analyzeButton.addEventListener("click", analyzeScan);

    newScanButton.addEventListener("click", () => {
        resetUpload();
        document.getElementById("workspace").scrollIntoView({ behavior: "smooth", block: "start" });
    });
}

async function loadModelInfo() {
    try {
        const response = await fetch("/api/model-info");
        if (!response.ok) {
            throw new Error("Model metadata request failed.");
        }

        const data = await response.json();
        modelReady = Boolean(data.model_loaded);
        renderModelStatus(data);
        renderClassCards(data.classes || []);
        renderArtifacts(data.artifacts || {});
        updateAnalyzeButton();
    } catch (error) {
        modelReady = false;
        modelStatus.dataset.state = "error";
        modelStatusText.textContent = error.message || "Unable to read model status.";
        updateAnalyzeButton();
    }
}

function renderModelStatus(data) {
    modelStatus.dataset.state = data.model_loaded ? "ready" : "error";
    modelStatusText.textContent = data.model_loaded
        ? "Model ready"
        : data.model_error || "Model is not loaded.";

    document.getElementById("classCount").textContent = String((data.classes || []).length || 4);
    document.getElementById("imageSize").textContent = Array.isArray(data.image_size)
        ? `${data.image_size[0]} x ${data.image_size[1]}`
        : "224 x 224";
    document.getElementById("modelFile").textContent = data.model_file || ".h5";

    const testAccuracy = data.model_metadata && typeof data.model_metadata.test_accuracy === "number"
        ? `${(data.model_metadata.test_accuracy * 100).toFixed(1)}%`
        : "--";
    document.getElementById("testAccuracy").textContent = testAccuracy;
}

function renderClassCards(classes) {
    classGrid.innerHTML = "";

    classes.forEach((item, index) => {
        const card = document.createElement("article");
        card.className = "class-card";

        const img = document.createElement("img");
        img.src = item.sample_url;
        img.alt = `${item.label} sample MRI`;
        img.loading = "lazy";

        const label = document.createElement("span");
        label.className = "class-label";
        label.textContent = item.key;

        const classIndex = document.createElement("span");
        classIndex.className = "class-index";
        classIndex.textContent = `Class ${String(index + 1).padStart(2, "0")}`;

        const meta = document.createElement("div");
        meta.className = "class-card-meta";
        meta.append(label, classIndex);

        const title = document.createElement("h3");
        title.textContent = item.label;

        const copy = document.createElement("p");
        copy.textContent = item.description;

        const body = document.createElement("div");
        body.className = "class-card-copy";
        body.append(title, copy);

        card.append(img, meta, body);
        classGrid.appendChild(card);
    });
}

function renderArtifacts(artifacts) {
    artifactGrid.querySelectorAll(".artifact-card").forEach((card) => {
        const artifactKey = card.dataset.artifact;
        const image = card.querySelector("img");
        const artifact = artifacts[artifactKey];

        if (artifact && artifact.available) {
            image.src = artifact.url;
            card.classList.remove("is-missing");
        } else {
            image.removeAttribute("src");
            card.classList.add("is-missing");
        }
    });
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add("dragover");
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove("dragover");
}

function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove("dragover");

    const file = event.dataTransfer.files[0];
    if (file) {
        processFile(file);
    }
}

function processFile(file) {
    const validTypes = ["image/png", "image/jpeg", "image/jpg"];

    if (!validTypes.includes(file.type)) {
        showError("Upload a PNG, JPG, or JPEG MRI image.");
        return;
    }

    if (file.size > 16 * 1024 * 1024) {
        showError("File size must be less than 16 MB.");
        return;
    }

    selectedFile = file;
    hideError();

    const reader = new FileReader();
    reader.onload = (event) => {
        previewImg.src = event.target.result;
        fileName.textContent = file.name;
        fileMeta.textContent = `${formatFileSize(file.size)} selected`;
        uploadArea.hidden = true;
        imagePreview.hidden = false;
        showAwaitingAnalysis();
        setWorkspaceStage("selected");
        updateAnalyzeButton();
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = "";
    previewImg.removeAttribute("src");
    fileName.textContent = "Selected image";
    fileMeta.textContent = "Ready for analysis";
    uploadArea.hidden = false;
    imagePreview.hidden = true;
    hideError();
    resetResults();
    setWorkspaceStage("empty");
    updateAnalyzeButton();
}

function resetResults() {
    resultsSection.dataset.state = "empty";
    delete resultsSection.dataset.risk;
    resultBadge.textContent = "No case";
    document.getElementById("resultDiagnosis").textContent = "Awaiting scan";
    document.getElementById("resultMessage").textContent = "The prediction summary will appear here after analysis.";
    document.getElementById("confidenceValue").textContent = "0%";
    document.getElementById("confidenceFill").style.width = "0%";
    document.getElementById("probabilityList").innerHTML = "";
    emptyResultText.textContent = "Select a scan to generate a result summary.";
    emptyResult.hidden = false;
    resultContent.hidden = true;
}

function showAwaitingAnalysis() {
    resultsSection.dataset.state = "selected";
    delete resultsSection.dataset.risk;
    resultBadge.textContent = "Ready";
    document.getElementById("resultDiagnosis").textContent = "Ready to analyze";
    document.getElementById("resultMessage").textContent = "The prediction summary will appear here after analysis.";
    document.getElementById("confidenceValue").textContent = "0%";
    document.getElementById("confidenceFill").style.width = "0%";
    document.getElementById("probabilityList").innerHTML = "";
    emptyResultText.textContent = "Run analysis to reveal the prediction, confidence, and probability spread.";
    emptyResult.hidden = false;
    resultContent.hidden = true;
}

function setWorkspaceStage(stage) {
    workspaceSection.dataset.stage = stage;
    resultsSection.setAttribute("aria-hidden", String(stage === "empty"));
}

function updateAnalyzeButton() {
    analyzeButton.disabled = !selectedFile || !modelReady;
}

async function analyzeScan() {
    if (!selectedFile || !modelReady) {
        return;
    }

    setLoading(true);
    hideError();

    try {
        const formData = new FormData();
        formData.append("file", selectedFile);

        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed.");
        }

        displayResults(data);
    } catch (error) {
        showError(error.message || "Failed to analyze scan. Please try again.");
    } finally {
        setLoading(false);
    }
}

function setLoading(isLoading) {
    analyzeButton.disabled = isLoading || !selectedFile || !modelReady;
    analyzeButton.querySelector(".button-text").hidden = isLoading;
    analyzeButton.querySelector(".button-loader").hidden = !isLoading;
    analyzeButton.querySelector(".button-loader").style.display = isLoading ? "flex" : "";

    if (isLoading) {
        setWorkspaceStage("loading");
        resultsSection.dataset.state = "loading";
        resultBadge.textContent = "Processing";
        document.getElementById("resultDiagnosis").textContent = "Processing scan";
        emptyResultText.textContent = "Running local inference and preparing probability evidence.";
        emptyResult.hidden = false;
        resultContent.hidden = true;
    } else if (resultsSection.dataset.state === "loading") {
        showAwaitingAnalysis();
        setWorkspaceStage(selectedFile ? "selected" : "empty");
    }
}

function displayResults(data) {
    const confidence = data.confidence || 0;
    const confidencePercent = Math.round(confidence * 1000) / 10;
    const isNoTumor = data.class_key === "notumor";
    const risk = getResultRisk(data.class_key, confidence);

    setWorkspaceStage("complete");
    resultsSection.dataset.state = "complete";
    resultsSection.dataset.risk = risk;
    emptyResult.hidden = true;
    resultContent.hidden = false;

    document.getElementById("resultDiagnosis").textContent = data.prediction;
    document.getElementById("resultMessage").textContent = getResultMessage(data.prediction, isNoTumor, confidence);
    document.getElementById("confidenceValue").textContent = `${confidencePercent.toFixed(1)}%`;

    resultBadge.textContent = getResultBadge(isNoTumor, confidence);

    const confidenceFill = document.getElementById("confidenceFill");
    confidenceFill.style.width = `${Math.min(confidencePercent, 100)}%`;
    confidenceFill.style.background = confidenceColor(confidence);

    renderProbabilities(data.all_probabilities || {});
    resultsSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderProbabilities(probabilities) {
    const probabilityList = document.getElementById("probabilityList");
    probabilityList.innerHTML = "";

    Object.entries(probabilities).forEach(([name, probability]) => {
        const percent = Math.max(0, Math.min(probability * 100, 100));
        const item = document.createElement("div");
        item.className = "probability-item";

        const main = document.createElement("div");
        main.className = "probability-main";

        const row = document.createElement("div");
        row.className = "probability-row";

        const label = document.createElement("span");
        label.className = "probability-name";
        label.textContent = name;

        const value = document.createElement("span");
        value.className = "probability-value";
        value.textContent = `${percent.toFixed(1)}%`;

        const bar = document.createElement("div");
        bar.className = "mini-bar";
        bar.setAttribute("aria-hidden", "true");

        const fill = document.createElement("div");
        fill.className = "mini-fill";
        fill.style.width = `${percent}%`;
        fill.style.background = confidenceColor(probability);

        row.append(label, value);
        bar.appendChild(fill);
        main.append(row, bar);
        item.appendChild(main);
        probabilityList.appendChild(item);
    });
}

function getResultRisk(classKey, confidence) {
    if (confidence < 0.7) {
        return "watch";
    }
    return classKey === "notumor" ? "clear" : "alert";
}

function getResultBadge(isNoTumor, confidence) {
    if (confidence < 0.7) {
        return "Low confidence";
    }
    return isNoTumor ? "No tumor" : "Tumor class";
}

function getResultMessage(prediction, isNoTumor, confidence) {
    if (confidence < 0.7) {
        return `Top model output is ${prediction}, but confidence is low. Review class spread before using this result.`;
    }

    if (isNoTumor) {
        return "Top model output is no tumor detected. Continue with clinical review before concluding.";
    }

    return `Top model output is ${prediction}. Compare probability spread before escalation.`;
}

function confidenceColor(confidence) {
    if (confidence >= 0.85) {
        return "#15803d";
    }
    if (confidence >= 0.7) {
        return "#b45309";
    }
    return "#b91c1c";
}

function formatFileSize(size) {
    if (size < 1024 * 1024) {
        return `${Math.max(1, Math.round(size / 1024))} KB`;
    }
    return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function showError(message) {
    errorText.textContent = message;
    errorMessage.hidden = false;
}

function hideError() {
    errorMessage.hidden = true;
}

function setupLightbox() {
    const lightbox = document.getElementById("imageLightbox");
    const lightboxClose = document.getElementById("lightboxClose");
    const lightboxImg = document.getElementById("lightboxImg");
    const lightboxCaption = document.getElementById("lightboxCaption");

    if (!lightbox || !lightboxClose || !lightboxImg || !lightboxCaption) {
        return;
    }

    function openLightbox(src, altText) {
        if (!src) return;
        lightboxImg.src = src;
        lightboxCaption.textContent = altText || "Image Preview";
        lightbox.removeAttribute("hidden");
        // Trigger reflow
        void lightbox.offsetWidth;
        lightbox.classList.add("active");
        document.body.style.overflow = "hidden";
    }

    function closeLightbox() {
        lightbox.classList.remove("active");
        document.body.style.overflow = "";
        setTimeout(() => {
            if (!lightbox.classList.contains("active")) {
                lightbox.setAttribute("hidden", "true");
                lightboxImg.src = "";
            }
        }, 300);
    }

    document.addEventListener("click", (e) => {
        if (e.target.tagName === "IMG" && e.target.src) {
            const isPreview = e.target.id === "previewImg";
            const isArtifact = e.target.closest(".artifact-card") && !e.target.closest(".artifact-card").classList.contains("is-missing");
            const isClassCard = e.target.closest(".class-card");

            if (isPreview || isArtifact || isClassCard) {
                let caption = e.target.alt;
                if (isArtifact) {
                    const cardTitle = e.target.closest(".artifact-card").querySelector("h3")?.textContent;
                    if (cardTitle) caption = `Model Evidence: ${cardTitle}`;
                } else if (isClassCard) {
                    const cardTitle = e.target.closest(".class-card").querySelector("h3")?.textContent;
                    if (cardTitle) caption = `Class Reference: ${cardTitle}`;
                } else if (isPreview) {
                    caption = `Uploaded MRI: ${fileName.textContent}`;
                }
                openLightbox(e.target.src, caption);
            }
        }
    });

    lightboxClose.addEventListener("click", closeLightbox);

    lightbox.addEventListener("click", (e) => {
        if (e.target === lightbox) {
            closeLightbox();
        }
    });

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && lightbox.classList.contains("active")) {
            closeLightbox();
        }
    });
}
