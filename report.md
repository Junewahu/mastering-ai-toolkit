# AI Tools and Applications Assignment Report

**Theme:** Mastering the AI Toolkit  
**Author:** [June Wahu]  
**Profession:** Medical Doctor, MBChB Student

---

## PART 1: THEORETICAL UNDERSTANDING

### 1. Short Answer Questions

**Q1: Differences between TensorFlow and PyTorch**

| Aspect              | TensorFlow                                    | PyTorch                                        |
| ------------------- | --------------------------------------------- | ---------------------------------------------- |
| Computational Graph | Static by default (uses `tf.Graph`)           | Dynamic (eager execution using Python control) |
| Syntax              | Verbose, less pythonic                        | More intuitive, pythonic                       |
| Deployment          | TensorFlow Serving, TensorFlow Lite supported | Requires additional tools (e.g., TorchServe)   |
| Preferred For       | Production-scale models, mobile deployment    | Research, prototyping, academic work           |

**When to Choose:**
- Use **TensorFlow** for production-ready deployment (especially mobile or embedded).
- Use **PyTorch** for fast prototyping, flexibility, and research.

**Q2: Use Cases of Jupyter Notebooks in AI Development**
1. Experimentation and Iteration: Easily test models step-by-step, visualize data, and inspect variables during model training.
2. Documentation and Sharing: Combine code, results, and explanations in one document for reproducible research and collaboration.

**Q3: spaCy vs Basic Python String Operations in NLP**
- **spaCy** provides advanced NLP components (NER, POS tagging, dependency parsing) out-of-the-box.
- **String Operations** (e.g., `.split()`, `.find()`) are simplistic and fail to handle linguistic context.
- **spaCy Advantages:** Accuracy, language model support, built-in pipelines, tokenisation respecting grammar and semantics.

### 2. Comparative Analysis: Scikit-learn vs TensorFlow

| Criteria          | Scikit-learn                                  | TensorFlow                                    |
| ----------------- | --------------------------------------------- | --------------------------------------------- |
| Focus             | Classical ML (e.g., decision trees, SVMs)     | Deep Learning (CNNs, RNNs, Transformers)      |
| Use for Beginners | Easier due to simpler syntax and API          | Steeper learning curve                        |
| Community Support | Strong, especially for ML education and tools | Massive, backed by Google, production-focused |

**Summary:**
Use **Scikit-learn** for tabular data and ML fundamentals. Use **TensorFlow** for complex tasks like image, text, or speech processing using neural networks.

---

## PART 2: PRACTICAL IMPLEMENTATION

### Task 1: Classical ML with Scikit-learn
- See `task1_classical_ml/iris_decision_tree.ipynb`

### Task 2: Deep Learning with TensorFlow
- See `task2_deep_learning/cnn_mnist_tensorflow.ipynb`
- ![Accuracy Graph](screenshots/accuracy_graph.png)
- ![Sample Predictions](task2_deep_learning/model_visuals/sample_predictions.png)

### Task 3: NLP with spaCy
- See `task3_nlp_spacy/amazon_reviews_ner_sentiment.ipynb`

---

## PART 3: ETHICS & OPTIMISATION

### 1. Ethical Reflection

**Bias in MNIST:**
- Overfitting to style of digits (e.g., Western-style '1').
- Underrepresentation of handwriting variations (left-handed users, cultural scripts).

**Mitigation Tools:**
- **TensorFlow Fairness Indicators:** Analyse per-group performance, identify disparities.
- **spaCy Rule-Based Matching:** Ensure product/brand names are culturally inclusive.

**Bias in Amazon Reviews:**
- Sentiment skewed by regional slang or sarcasm.
- Gender or product-category-based bias.

**Solutions:**
- Custom sentiment lexicons.
- Bias-aware preprocessing pipelines.

### 2. Troubleshooting Challenge: Fix TensorFlow Code

**Original Bug Example:**
```python
model.compile(loss='binary_crossentropy')  # Wrong for 10-class problem
```
**Fix:**
```python
model.compile(loss='sparse_categorical_crossentropy')
```

**Error: Dimension mismatch in labels**
```python
y_train = tf.keras.utils.to_categorical(y_train, 10)  # Incorrect if using sparse loss
```
**Fix:** Use raw integer labels for sparse loss, or switch loss function to `categorical_crossentropy`.

---

## BONUS TASK – Deployment via Streamlit

- See `bonus_deployment/streamlit_app.py`
- ![Streamlit UI](screenshots/streamlit_ui.png)

---

## GitHub Submission Checklist
- [x] `task1_classical_ml/iris_decision_tree.ipynb`
- [x] `task2_deep_learning/cnn_mnist_tensorflow.ipynb`
- [x] `task3_nlp_spacy/amazon_reviews_ner_sentiment.ipynb`
- [x] `bonus_deployment/streamlit_app.py`
- [x] `report.pdf`
- [x] `README.md`
- [x] `screenshots/` 