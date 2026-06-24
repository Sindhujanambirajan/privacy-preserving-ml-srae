#  Privacy-Preserving ML System
## Multi-Objective Supervised Residual Autoencoder

---

## Project Structure

```
privacy_ml_project/
├── backend/
│   ├── model.py        → SRAE Architecture
│   ├── losses.py       → 4 Loss Functions
│   ├── train.py        → Training Pipeline
│   ├── evaluate.py     → Performance Evaluation
│   ├── utils.py        → Helper Functions
│   └── app.py          → Flask Web Server
│
├── frontend/
│   └── index.html      → Web Interface
│
├── data/               → Put your datasets here
├── saved_models/       → Trained models saved here
├── results/            → Plots and metrics
└── notebooks/
    └── Privacy_ML_Training.ipynb  → Google Colab
```

---

## HOW TO RUN — Step by Step

### STEP 1: Install Requirements (VS Code Terminal)
```bash
pip install tensorflow==2.5.0 keras scikit-learn numpy pandas
pip install matplotlib seaborn flask flask-cors
```

### STEP 2: Train Model on Google Colab
1. Open: https://colab.research.google.com
2. Upload: `notebooks/Privacy_ML_Training.ipynb`
3. Runtime → Change Runtime → GPU
4. Upload your CSV dataset
5. Run all cells
6. Download `srae_best.h5` → Put in `saved_models/`

### STEP 3: Start Flask Server (VS Code Terminal)
```bash
cd backend
python app.py
```
Server runs at: http://localhost:5000

### STEP 4: Open Frontend
Open `frontend/index.html` in Chrome browser
OR go to: http://localhost:5000

---

##  CSV Format
Your CSV file should be:
- All feature columns first
- Last column = class label

Example (leukemia.csv):
```
gene1, gene2, gene3, ..., cancer_type
0.12,  0.45,  0.78,  ..., ALL
0.23,  0.56,  0.89,  ..., AML
```

---

##  Testing with Demo Data
```python
# In VS Code terminal:
python backend/utils.py  # loads MNIST automatically
```

## 🔐 Privacy Guarantee
- Original data NEVER leaves your system
- Only encoding Ψ (480 dims) is shared
- Without the private encoder, Ψ is unreadable
- Safe against 3 out of 4 attack scenarios
