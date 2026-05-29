import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY

data_dir = r"c:\Users\GS Adithya Krishna\Desktop\S6\NLP\offline-ai-tutor\data"
os.makedirs(data_dir, exist_ok=True)

topics = {
    "Neural_Networks_Basics.pdf": """NEURAL NETWORKS BASICS

A neural network is a machine learning model inspired by the structure and function of the human brain. It consists of interconnected layers of artificial neurons, also known as nodes or perceptrons.

1. Architecture
A typical neural network contains three types of layers:
- Input Layer: Receives the raw data (e.g., pixel values of an image).
- Hidden Layers: One or more layers that perform computations and extract features. Deep learning refers to networks with many hidden layers.
- Output Layer: Produces the final prediction or classification.

2. Weights and Biases
Each connection between neurons has an associated weight, which determines the importance of the input value. A bias term is also added to shift the activation function. During training, the network adjusts these weights and biases to minimize the error in its predictions.

3. Activation Functions
An activation function is applied to the weighted sum of inputs at each neuron to introduce non-linearity. Without non-linearity, a neural network, no matter how deep, would only be able to learn linear relationships. Common activation functions include:
- ReLU (Rectified Linear Unit): Outputs the input if positive, otherwise zero. It is the most popular activation function for hidden layers.
- Sigmoid: S-shaped curve that outputs a value between 0 and 1. Often used for binary classification.
- Softmax: Used in the output layer for multi-class classification to convert raw scores into probabilities.

4. Backpropagation
Backpropagation (backward propagation of errors) is the primary algorithm used to train neural networks. After a forward pass generates a prediction, the error is calculated using a loss function (e.g., Mean Squared Error or Cross-Entropy). Backpropagation then uses the chain rule of calculus to compute the gradient of the loss function with respect to each weight. An optimization algorithm, like Stochastic Gradient Descent (SGD) or Adam, uses these gradients to update the weights in the direction that minimizes the loss.
""",
    
    "Photosynthesis_Process.pdf": """THE PROCESS OF PHOTOSYNTHESIS

Photosynthesis is the fundamental biological process by which green plants, algae, and some bacteria convert light energy into chemical energy. This energy is stored in the bonds of glucose (a simple sugar) and other organic molecules, which serve as the primary food source for almost all life on Earth.

1. The Chemical Equation
The overall balanced chemical equation for photosynthesis is:
6CO2 + 6H2O + Light Energy -> C6H12O6 + 6O2

This means that six molecules of carbon dioxide and six molecules of water, in the presence of light, produce one molecule of glucose and six molecules of oxygen gas as a byproduct.

2. Chloroplasts and Chlorophyll
In plants, photosynthesis occurs primarily in the leaves, specifically within specialized organelles called chloroplasts. Chloroplasts contain chlorophyll, the green pigment responsible for capturing light energy. The internal structure of a chloroplast includes the stroma (the fluid-filled space) and the thylakoids (membranous sacs arranged in stacks called grana).

3. The Light-Dependent Reactions
The first stage of photosynthesis takes place in the thylakoid membrane and requires direct light. 
- Chlorophyll absorbs photons of light, exciting electrons to a higher energy state.
- Water molecules are split (photolysis) to replace these excited electrons, releasing oxygen gas (O2) into the atmosphere.
- The excited electrons travel through an electron transport chain, generating ATP (adenosine triphosphate) and NADPH (nicotinamide adenine dinucleotide phosphate). These two molecules act as energy carriers for the next stage.

4. The Calvin Cycle (Light-Independent Reactions)
The second stage occurs in the stroma and does not require direct light, although it relies on the ATP and NADPH produced in the first stage.
- Carbon Fixation: The enzyme RuBisCO captures CO2 from the atmosphere and attaches it to a 5-carbon sugar called RuBP.
- Reduction: The ATP and NADPH are used to convert the resulting 6-carbon compound into molecules of G3P (glyceraldehyde 3-phosphate), a simple sugar.
- Regeneration: Some G3P molecules are used to build glucose, while others are recycled to regenerate RuBP, allowing the cycle to continue.
""",
    
    "Quantum_Computing_Intro.pdf": """QUANTUM COMPUTING: AN INTRODUCTION

Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers. While traditional computers use binary bits, quantum computers use quantum bits, or qubits.

1. Superposition
In classical computing, a bit must be in one of two states: 0 or 1. A qubit, however, can exist in a state of 0, 1, or any quantum superposition of these two states. This means a qubit can represent both 0 and 1 simultaneously. When a quantum computer has multiple qubits in superposition, it can process a vast number of possibilities at the same time, giving it an exponential computational advantage for certain types of problems.

2. Entanglement
Entanglement is a quantum phenomenon where two or more qubits become linked in such a way that the state of one qubit instantly influences the state of the other, regardless of the distance separating them. In a quantum computer, entangled qubits allow for complex correlations between variables, enabling the system to evaluate multiple interconnected paths simultaneously. 

3. Quantum Interference
Just like waves in a pond can constructively build upon each other or destructively cancel each other out, quantum states can exhibit interference. Quantum algorithms are designed to use interference to amplify the probability of the correct answer and cancel out the probabilities of incorrect answers, eventually collapsing the superposition into the correct classical output when measured.

4. Applications and Challenges
Quantum computing has the potential to revolutionize several fields:
- Cryptography: Shor's algorithm could break widely used encryption schemes (like RSA), prompting the development of post-quantum cryptography.
- Drug Discovery: Simulating molecular interactions at the quantum level could drastically reduce the time needed to design new pharmaceuticals.
- Optimization: Solving complex logistical problems, such as supply chain routing and portfolio management.

The primary challenge in building quantum computers is maintaining "decoherence." Qubits are extremely sensitive to their environment (heat, electromagnetic fields). Any disturbance causes them to lose their quantum state, leading to errors. To combat this, most modern quantum computers must be cooled to near absolute zero (-273 degrees Celsius).
"""
}

styles = getSampleStyleSheet()
styleN = styles["BodyText"]
styleN.alignment = TA_JUSTIFY
styleH = styles["Heading1"]

for filename, content in topics.items():
    output_path = os.path.join(data_dir, filename)
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    Story = []
    
    paragraphs = content.split('\n\n')
    for idx, p in enumerate(paragraphs):
        p = p.replace('\n', ' ')
        if idx == 0:
            Story.append(Paragraph(p, styleH))
        else:
            Story.append(Paragraph(p, styleN))
        Story.append(Spacer(1, 12))
        
    doc.build(Story)
    print(f"Created {output_path}")

print("Done!")
