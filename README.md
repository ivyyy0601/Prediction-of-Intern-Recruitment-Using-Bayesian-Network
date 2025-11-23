# 📚 Character Relationship Network for Detective Fiction  
A NLP + Graph Analysis project to uncover hidden relationships, plot structures, and character importance across multiple detective novels.

This project uses **Named Entity Recognition**, **Sentiment Analysis**, **Co-occurrence Graphs**, and **PageRank** to automatically build **character relationship networks** from raw novel text.  
It helps readers quickly understand plot progression, character evolution, and emotional tone.

---

## ✨ Features
- ✔ Automatic character extraction (spaCy NER)  
- ✔ Sentence-level sentiment scoring (Afinn)  
- ✔ Co-occurrence matrix computation  
- ✔ Sentiment matrix with alignment rate  
- ✔ Network graph visualization (NetworkX + Matplotlib)  
- ✔ PageRank-based key character detection  
- ✔ Multi-book batch processing  
- ✔ High-resolution PNG graph outputs  

---

## 🧱 1. Project Workflow

### **1️⃣ Read and preprocess novel text**
```python
novel = read_novel(file_path)
sentence_list = sent_tokenize(novel)
```

### **2️⃣ Compute sentiment alignment rate**
Ensures consistency across authors and writing styles.
```python
align_rate = calculate_align_rate(sentence_list)
```

### **3️⃣ Named Entity Recognition (NER)**
Extracts all PERSON and ORG names, applies filtering & normalization.
```python
preliminary_names = iterative_NER(sentence_list)
```

### **4️⃣ Determine top characters via frequency**
```python
name_frequency, name_list = top_names(preliminary_names, novel, 25)
```

### **5️⃣ Compute matrices**
```python
co_matrix, sentiment_matrix = calculate_matrix(name_list, sentence_list, align_rate)
```

### **6️⃣ Visualize network graphs**
```python
plot_graph(name_list, name_frequency, co_matrix, 'co-occurrence', 'co-occurrence')
plot_graph(name_list, name_frequency, sentiment_matrix, 'sentiment', 'sentiment')
```

### **7️⃣ PageRank ranking of characters**
```python
top_pr = top_names_with_pagerank(co_matrix, name_list, 5)
print(top_pr)
```

---

## 🔍 2. Key Algorithms

### **Named Entity Recognition**
- Based on spaCy `en_core_web_sm`
- Splits multi-word names
- Removes common English words
- Removes tokens < 3 letters
- Deduplicates and filters noise based on frequency threshold

### **Sentiment Analysis**
- Uses Afinn scoring  
- Alignment rate adjusts sentiment skew  

### **Matrices**
- Co-occurrence = name occurrence × transpose  
- Sentiment = co-occurrence × sentence sentiment  
- Both triangularized & normalized

### **PageRank**
Ranks influence across the network graph using:
```python
nx.pagerank(G)
```

---

## 📊 3. Outputs

### ✔ Co-occurrence Network (PNG)  
Shows intensity of shared sentences.

### ✔ Sentiment Network (PNG)  
Edge color = friendliness vs hostility  
Node size = importance  

### ✔ PageRank Top Characters  
Example:
```
['sherlock', 'watson', 'poirot', 'hastings', 'villain']
```

---

## 📁 4. Project Structure

```
📁 character-network/
│── novels/
│── graphs/
│── common_words.txt
│── main.py
│── README.md
```

---

## 🛠 5. Installation

### Install dependencies
```bash
pip install spacy afinn nltk networkx matplotlib pandas numpy
```

### Download spaCy model
```bash
python -m spacy download en_core_web_sm
```

---



## 🌟 6. Future Improvements
- Add coreference resolution (“he”, “she”, “I” → character names)  
- Transformer-based sentiment model (BERT/RoBERTa)  
- Interactive web graph (D3.js or PyVis)  
- Chapter-wise dynamic relationship evolution  
- Integration with text summarization model (T5 + PageRank)  

---

## 🎉 7. Summary
This project provides a full pipeline for analyzing story structure through:
- NLP  
- Graph theory  
- Sentiment modeling  
- PageRank centrality  

It is designed for detective fiction but can be applied to *any novel* with minimal changes.
