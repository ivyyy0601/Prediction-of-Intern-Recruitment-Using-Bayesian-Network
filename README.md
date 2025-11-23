# 📌 Prediction of Intern Recruitment Using Bayesian Networks

This project builds a Bayesian Network to predict the probability that a candidate will receive an internship admission offer from a tech company. The system models relationships among key factors such as education level, interview performance, salary, and welfare, and performs inference using Variable Elimination and the Clique Tree (Junction Tree) Algorithm.

## 🚀 1. Overview
The goal of this project is to estimate:

**P(Admission | candidate attributes, company preference, job conditions)**

The Bayesian Network integrates both candidate-side factors (e.g., interest, major relevance) and company-side factors (e.g., offer decision, interview performance), forming a comprehensive prediction model for internship admission.

## 🧱 2. Bayesian Network Construction

### **2.1 Nodes (12 Variables)**
The network includes 12 key factors:
- Education Level  
- Work Experience  
- Age  
- Interview Performance  
- Company Offer  
- Offer Accepted  
- Major Related  
- Working Hours  
- Interest  
- Salary  
- Welfare  
- Admission (target)

### **2.2 Logic Behind the Network**
- **Company Offer** is influenced by: Education Level, Interview Performance, Age, Work Experience  
- **Offer Accepted** is influenced by: Salary, Welfare, Interest, Working Hours, Major Related  
- **Admission** depends on both: Company Offer + Offer Accepted  

The network is represented as a Directed Acyclic Graph (DAG) with conditional probability tables derived from a job description dataset and supplemental research data.

---

## 🔍 3. Inference Methods
This project implements two exact inference algorithms:

### **3.1 Variable Elimination (VE)**
VE simplifies inference through:
- Selecting a variable to eliminate  
- Combining factors involving that variable  
- Marginalizing (summing out)  
- Repeating until only query variables remain  

Two implementations:
- pgmpy-based VE  
- Custom VE (factor initialization, combination, marginalization, normalization)

### **3.2 Clique Tree / Junction Tree Algorithm**
Steps:
1. Moralization & Triangulation  
2. Convert DAG → Junction Tree  
3. Message passing  
4. Clique potential updates  
5. Compute marginal/conditional probabilities  

Suitable for dense networks with many dependencies.

---

## 🖥 4. User Interface (UI)
The system includes a simple UI that allows users to:
- Input conditions (e.g., “Master’s degree”, “Good interview performance”)  
- View predicted admission probability  
- Edit CPTs and observe changes  

---

## 📊 5. Experimental Results

### **5.1 Predictive Inference**
- **Education**  
  - Bachelor: 0.6149  
  - Master: 0.6416  

- **Age > 30**  
  - Probability improves with work experience and stable working hours  

- **Welfare**  
  - Good: 0.5914  
  - Average: 0.4086  

- **Working Hours**  
  - Best: 5–8 hours/day (0.4962)

### **5.2 Diagnostic Inference**
Used to understand:  
- Factors driving acceptance/rejection  
- “What-if” scenario analysis  

---

## ⚖️ 6. Method Comparison

| Method | Strength | Weakness | Best Use Case |
|--------|----------|-----------|----------------|
| Variable Elimination | Fast, simple | Not ideal for large, dense networks | Small/medium BN |
| Clique Tree | Handles strong dependencies | Higher memory & computation | Large, complex BN |

---

## 🧩 7. Conclusion
Bayesian Networks provide:
- Transparent causal reasoning  
- Flexible simulation  
- Evidence-based predictions  
- Easy integration with UI  

Future work may include automatic structure learning, additional datasets, and web deployment.

---

## 🛠 8. How to Run

### Install dependencies:
```bash
pip install pgmpy numpy pandas
```

### Run Bayesian Network:
```python
from pgmpy.inference import VariableElimination

inference = VariableElimination(model)
result = inference.query(
    variables=["Admission"],
    evidence={"Education": "Master", "Interview": "Good"}
)
print(result)
```

### Launch UI:
```bash
python ui.py
```
