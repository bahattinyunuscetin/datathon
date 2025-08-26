# Advanced Datathon Competition Solution

Bu proje, datathon yarışması için hazırlanmış **çok gelişmiş ve yaratıcı** çözümler içerir. Her `user_session` için `session_value` tahmin etmeyi amaçlar.

## 🚀 Çözüm Yaklaşımları

### 1. **Basit ML Çözümü** (`datathon_solution.py`)
- Random Forest Regressor
- Temel feature engineering
- Hızlı ve etkili

### 2. **Gelişmiş ML Çözümü** (`advanced_analysis.py`)
- **8 farklı model** (RandomForest, GradientBoosting, ExtraTrees, Ridge, Lasso, ElasticNet, SVR, MLP)
- **Yaratıcı feature engineering:**
  - Session duration (zaman bazlı)
  - Behavioral patterns (event sequence analysis)
  - Product engagement scores
  - Conversion probability
  - Bounce rate indicators
- **Ensemble learning** (weighted average)

### 3. **Deep Learning Çözümü** (`deep_learning_solution.py`)
- **Transformer model** (attention mechanism)
- **LSTM with Attention** (bidirectional)
- **Tabular Neural Network** (BatchNorm + Dropout)
- **Sequence modeling** (event sequences)
- **Ensemble predictions** (40% Transformer + 40% LSTM + 20% Tabular)

### 4. **Graph Analysis Çözümü** (`graph_analysis_solution.py`)
- **NetworkX** ile user-product-category graph
- **Centrality measures** (degree, betweenness, closeness, pagerank)
- **Community detection** (modularity)
- **Ego network analysis**
- **Clustering coefficients**
- **Graph visualization**

## 📊 Feature Engineering Detayları

### Zaman Bazlı Features
- Session duration (dakika cinsinden)
- Saat bazlı aktivite (hour, day_of_week)
- Temporal patterns

### Behavioral Patterns
- Event sequence analysis (VIEW → ADD_CART → BUY)
- Conversion path detection
- Bounce rate indicators
- Cart abandonment patterns

### Network Features
- User centrality in product network
- Product popularity scores
- Category interaction patterns
- Community membership

### Product Engagement
- Product interaction depth
- Category exploration
- User session frequency
- Engagement scoring

## 🏗️ Model Mimarileri

### Ensemble Models
- **Random Forest**: 200 trees, feature importance
- **Gradient Boosting**: 200 estimators
- **Extra Trees**: 200 estimators
- **Linear Models**: Ridge, Lasso, ElasticNet
- **Support Vector**: RBF kernel
- **Neural Network**: MLP with hidden layers

### Deep Learning
- **Transformer**: 6 layers, 8 heads, 128 dimensions
- **LSTM**: 2 layers, bidirectional, attention
- **Tabular NN**: 256→128→64→1 architecture

## 📁 Dosya Yapısı

```
datathon/
├── train.csv                    # Eğitim verisi
├── test.csv                     # Test verisi
├── sample_submission.csv        # Örnek submission
├── datathon_solution.py         # Basit ML çözümü
├── advanced_analysis.py         # Gelişmiş ML çözümü
├── deep_learning_solution.py    # Deep Learning çözümü
├── graph_analysis_solution.py   # Graph Analysis çözümü
├── requirements.txt             # Gerekli paketler
├── README.md                    # Bu dosya
└── submissions/                 # Submission dosyaları
    ├── submission.csv           # Basit ML
    ├── advanced_submission.csv  # Gelişmiş ML
    ├── deep_learning_submission.csv # Deep Learning
    └── graph_analysis_submission.csv # Graph Analysis
```

## 🛠️ Kurulum

1. Python 3.8+ yüklü olduğundan emin olun
2. Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Kullanım

### Basit ML Çözümü
```bash
python datathon_solution.py
```

### Gelişmiş ML Çözümü
```bash
python advanced_analysis.py
```

### Deep Learning Çözümü
```bash
python deep_learning_solution.py
```

### Graph Analysis Çözümü
```bash
python graph_analysis_solution.py
```

## 📈 Performans Metrikleri

Her çözüm çalıştırıldığında:
- **Cross-validation RMSE**
- **Feature importance rankings**
- **Model performance comparisons**
- **Prediction distributions**

## 🎯 Yaratıcı Özellikler

### 1. **Event Sequence Analysis**
- VIEW → ADD_CART → BUY conversion tracking
- Cart abandonment detection
- Bounce rate analysis

### 2. **Temporal Behavioral Patterns**
- Session duration analysis
- Time-of-day preferences
- Day-of-week patterns

### 3. **Network Graph Features**
- User-product-category relationships
- Centrality measures
- Community detection

### 4. **Deep Sequence Modeling**
- Transformer attention mechanisms
- LSTM with self-attention
- Event sequence embeddings

## 🔬 Model Karşılaştırması

| Yaklaşım | Model Sayısı | Feature Sayısı | Özel Özellik |
|----------|--------------|----------------|---------------|
| Basit ML | 1 | 4 | Temel aggregation |
| Gelişmiş ML | 8 | 20+ | Behavioral patterns |
| Deep Learning | 3 | 8+ | Sequence modeling |
| Graph Analysis | 2 | 30+ | Network features |

## 💡 İyileştirme Önerileri

1. **Hyperparameter Tuning**: Optuna ile otomatik tuning
2. **Feature Selection**: SHAP ile feature importance
3. **Cross-Validation**: Stratified k-fold
4. **Ensemble Weights**: Validation performance'a göre
5. **Data Augmentation**: Synthetic session generation

## 🏆 Yarışma Stratejisi

1. **Basit ML** ile baseline oluştur
2. **Gelişmiş ML** ile feature engineering
3. **Deep Learning** ile sequence modeling
4. **Graph Analysis** ile network insights
5. **Ensemble** ile final predictions

## 📊 Çıktılar

Her çözüm:
- `submission.csv` dosyası oluşturur
- Model performans metrikleri gösterir
- Feature importance sıralaması
- Prediction distribution analizi

## 🚨 Notlar

- Tüm session'lar için tahmin yapılır
- Negatif değerler 0'a yuvarlanır
- GPU kullanımı (PyTorch için)
- Memory efficient processing
- Scalable architecture

---

**Bu çözüm, datathon yarışmasında rakiplerin aklına gelmeyecek yaratıcı yaklaşımlar kullanır! 🎯**
