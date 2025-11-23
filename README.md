# Prediction of Intern Recruitment Using Bayesian Networks

This project builds a Bayesian Network to predict the probability that a candidate will receive an internship admission offer from a tech company. The system models relationships among key factors such as education level, interview performance, salary, and welfare, and performs inference using Variable Elimination and the Clique Tree (Junction Tree) Algorithm.

## 1. Overview
The goal of this project is to estimate:
**P(Admission | candidate attributes, company preference, job conditions)**

## 2. Bayesian Network Construction
- 12 variables modeled (Education Level, Work Experience, Age, Interview Performance, Salary, Welfare, Company Offer, Offer Accepted, Major Related, Working Hours, Interest, Admission)
- DAG structure defined based on hiring logic
- CPTs derived from datasets and assumptions

## 3. Inference Methods
### Variable Elimination (VE)
- Eliminates irrelevant variables
- Combines factors and marginalizes variables
- Implemented via pgmpy and custom code

### Clique Tree Algorithm
- Moralization, triangulation, clique construction
- Message passing for efficient inference

## 4. User Interface
Interactive input for candidate attributes and real-time probability updates. CPTs are editable to observe outcome changes.

## 5. Results
- Master's admission probability: 0.6416
- Bachelor's admission probability: 0.6149
- Good welfare significantly increases acceptance
- Optimal working hours for highest admission probability: 5–8 hours/day

## 6. Comparison
| Method | Strength | Weakness |
|--------|----------|-----------|
| Variable Elimination | Fast, simple | Not ideal for dense networks |
| Clique Tree | Good for complex dependencies | Higher memory usage |

## 7. How to Run
Install dependencies:
```
pip install pgmpy numpy pandas
```

Run inference:
```
from pgmpy.inference import VariableElimination
inference = VariableElimination(model)
result = inference.query(variables=["Admission"], evidence={"Education": "Master"})
print(result)
```

Launch UI:
```
python ui.py
```
