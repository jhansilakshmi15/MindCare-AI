/**
 * MindCare AI - Client-Side Interactive Logic
 * Handles multi-step form navigation, validation, dynamic review generation,
 * interactive card selections, and PDF report printing.
 */

document.addEventListener("DOMContentLoaded", () => {
    // 1. Radio Card Selection Styling
    const optionCards = document.querySelectorAll(".option-select-card");
    optionCards.forEach(card => {
        const radio = card.querySelector("input[type='radio']");
        if (radio) {
            if (radio.checked) {
                card.classList.add("selected");
            }
            card.addEventListener("click", () => {
                const groupName = radio.name;
                document.querySelectorAll(`input[name="${groupName}"]`).forEach(r => {
                    const parentCard = r.closest(".option-select-card");
                    if (parentCard) parentCard.classList.remove("selected");
                });
                radio.checked = true;
                card.classList.add("selected");
                // Trigger change event for listeners
                radio.dispatchEvent(new Event("change"));
            });
        }
    });

    // 2. Stress Slider Live Value Display
    const stressRange = document.getElementById("stress_level");
    const stressDisplay = document.getElementById("stress_val_display");
    if (stressRange && stressDisplay) {
        stressRange.addEventListener("input", (e) => {
            stressDisplay.textContent = e.target.value;
        });
    }

    // 3. Multi-Step Form Navigation
    const form = document.getElementById("questionnaireForm");
    if (form) {
        let currentStep = 1;
        const totalSteps = 5;

        const progressBar = document.getElementById("stepperProgressBar");
        const stepNodes = document.querySelectorAll(".step-node");
        const stepPanels = document.querySelectorAll(".form-step-panel");

        const updateStepperUI = () => {
            // Update progress bar percentage
            const percentage = ((currentStep - 1) / (totalSteps - 1)) * 100;
            if (progressBar) progressBar.style.width = `${percentage}%`;

            // Update step nodes
            stepNodes.forEach((node, idx) => {
                const stepNum = idx + 1;
                node.classList.remove("active", "completed");
                if (stepNum === currentStep) {
                    node.classList.add("active");
                } else if (stepNum < currentStep) {
                    node.classList.add("completed");
                }
            });

            // Display active panel
            stepPanels.forEach(panel => {
                const step = parseInt(panel.getAttribute("data-step"), 10);
                if (step === currentStep) {
                    panel.classList.remove("d-none");
                } else {
                    panel.classList.add("d-none");
                }
            });

            // Populate Review on Step 5
            if (currentStep === 5) {
                populateReviewSummary();
            }

            // Scroll to top of questionnaire card smoothly
            const topAnchor = document.getElementById("questionnaireCard");
            if (topAnchor) {
                topAnchor.scrollIntoView({ behavior: "smooth", block: "start" });
            }
        };

        const validateStep = (step) => {
            if (step === 1) {
                const ageInput = document.getElementById("ageInput");
                const ageError = document.getElementById("ageError");
                const age = parseInt(ageInput.value, 10);

                if (isNaN(age) || age < 18 || age > 100) {
                    if (ageError) {
                        ageError.textContent = "Please enter a valid age between 18 and 100 years.";
                        ageError.classList.remove("d-none");
                    }
                    ageInput.focus();
                    return false;
                } else {
                    if (ageError) ageError.classList.add("d-none");
                }

                // Check Gender
                const genderChecked = document.querySelector("input[name='gender']:checked");
                if (!genderChecked) {
                    alert("Please select a gender option (or choose 'Prefer not to say').");
                    return false;
                }
            }
            return true;
        };

        const populateReviewSummary = () => {
            const getVal = (name) => {
                const el = document.querySelector(`input[name="${name}"]:checked`);
                return el ? el.value : "Not specified";
            };

            const age = document.getElementById("ageInput")?.value || "N/A";
            const gender = getVal("gender");
            const familyHistory = getVal("family_history");
            const workInterfere = getVal("work_interfere");
            const leave = getVal("leave");
            const benefits = getVal("benefits");
            const careOptions = getVal("care_options");
            const anonymity = getVal("anonymity");
            const stress = document.getElementById("stress_level")?.value || "5";
            const sleep = getVal("sleep_quality");
            const workload = getVal("workload");

            const reviewContainer = document.getElementById("reviewContentArea");
            if (reviewContainer) {
                reviewContainer.innerHTML = `
                    <div class="table-responsive">
                        <table class="table table-sm align-middle mb-0">
                            <thead class="table-light">
                                <tr>
                                    <th>Category</th>
                                    <th>Attribute</th>
                                    <th>Selected Response</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td rowspan="3" class="fw-bold text-muted bg-light">Demographics</td>
                                    <td>Age</td>
                                    <td><span class="badge bg-secondary">${age} years</span></td>
                                </tr>
                                <tr>
                                    <td>Gender Identity</td>
                                    <td><span class="badge bg-secondary">${gender}</span></td>
                                </tr>
                                <tr>
                                    <td>Family History</td>
                                    <td><span class="badge bg-secondary">${familyHistory}</span></td>
                                </tr>
                                <tr>
                                    <td rowspan="2" class="fw-bold text-muted bg-light">Work/Study Impact</td>
                                    <td>Interference with Activity</td>
                                    <td><span class="badge bg-primary">${workInterfere}</span></td>
                                </tr>
                                <tr>
                                    <td>Ease of Mental Health Leave</td>
                                    <td><span class="badge bg-primary">${leave}</span></td>
                                </tr>
                                <tr>
                                    <td rowspan="3" class="fw-bold text-muted bg-light">Support Resources</td>
                                    <td>Care Options Available</td>
                                    <td><span class="badge bg-info text-dark">${careOptions}</span></td>
                                </tr>
                                <tr>
                                    <td>Mental Health Benefits</td>
                                    <td><span class="badge bg-info text-dark">${benefits}</span></td>
                                </tr>
                                <tr>
                                    <td>Protected Anonymity</td>
                                    <td><span class="badge bg-info text-dark">${anonymity}</span></td>
                                </tr>
                                <tr>
                                    <td rowspan="3" class="fw-bold text-muted bg-light">Wellness Context (Optional)</td>
                                    <td>Self-Reported Stress (1-10)</td>
                                    <td><span class="badge bg-warning text-dark">${stress} / 10</span></td>
                                </tr>
                                <tr>
                                    <td>Sleep Quality</td>
                                    <td><span class="badge bg-warning text-dark">${sleep}</span></td>
                                </tr>
                                <tr>
                                    <td>Current Workload</td>
                                    <td><span class="badge bg-warning text-dark">${workload}</span></td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                `;
            }
        };

        // Next / Prev Button Listeners
        document.querySelectorAll(".btn-next-step").forEach(btn => {
            btn.addEventListener("click", () => {
                if (validateStep(currentStep)) {
                    if (currentStep < totalSteps) {
                        currentStep++;
                        updateStepperUI();
                    }
                }
            });
        });

        document.querySelectorAll(".btn-prev-step").forEach(btn => {
            btn.addEventListener("click", () => {
                if (currentStep > 1) {
                    currentStep--;
                    updateStepperUI();
                }
            });
        });

        // Initialize on load
        updateStepperUI();
    }
});

// Function to print/save result page as PDF
function printReport() {
    window.print();
}

