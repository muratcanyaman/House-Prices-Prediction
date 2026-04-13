# House Prices — Advanced Regression Techniques

Ames, Iowa'daki konut satış fiyatlarını tahmin etmeye yönelik Kaggle yarışması çalışması.

**Yarışma:** [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## Veri Seti

- **Train:** 1460 örnek, 79 özellik + hedef değişken (SalePrice)
- **Test:** 1459 örnek, 79 özellik
- Özellikler; alan, kalite puanları, yapım yılı, mahalle, garaj/bodrum detayları gibi mülk bilgilerini içermektedir.

## Yaklaşım

### Keşifsel Veri Analizi (EDA)

- Hedef değişken dağılım analizi ve log dönüşümü
- 35+ özellik için eksik veri profilleme
- Korelasyon analizi ve en güçlü tahmin ediciler için heatmap
- Sayısal ve kategorik özellikler için scatter plot ve boxplot görselleştirmeleri
- GrLivArea, TotalBsmtSF ve LotArea üzerinde aykırı değer tespiti

### Özellik Mühendisliği

- **Alan özellikleri:** TotalSF, TotalLivArea, TotalPorchSF, AvgRoomSize, BsmtFinRatio vb.
- **Yaş özellikleri:** HouseAge, RemodAge, GarageAge, IsNew, IsRemodeled
- **Kalite skorları:** 15+ kalite/durum sütunu için ordinal encoding, bileşik skorlar (OverallScore, TotalQualScore)
- **Etkileşim terimleri:** Kalite × Alan çarpımları, polinom terimler (GrLivArea², TotalSF², OverallQual²)
- **İkili göstergeler:** HasPool, HasGarage, HasBsmt, HasFireplace, Has2ndFloor vb.
- **Hedef encoding:** Mahalle bazlı ortalama hedef encoding
- **Çarpıklık düzeltme:** |skewness| > 0.75 olan özelliklere log1p uygulanması
- **Nihai özellik sayısı:** 257 (one-hot encoding sonrası)

### Modelleme

Hiperparametreleri ayarlanmış yedi model kullanılmıştır:

| Model | CV MAE ($) |
|-------|-----------|
| GradientBoosting | 13,295 |
| ElasticNet | 13,516 |
| Lasso | 13,559 |
| Ridge | 13,602 |
| CatBoost | 13,666 |
| XGBoost | 13,734 |
| LightGBM | 14,398 |

### Ensemble

- 5-Fold CV ile Out-of-Fold (OOF) tahminlerinin hesaplanması
- Nelder-Mead optimizasyonu (scipy.optimize) ile blend ağırlıklarının belirlenmesi
- Baskın katkı sağlayan modeller: ElasticNet (0.41) ve GradientBoosting (0.43)

## Sonuçlar

- **OOF MAE:** $12,700
- **Sıralama:** İlk 30

## Dosyalar

- `house-prices-advanced-regression.ipynb` — Tam notebook (EDA + modelleme + submission)
- `train.csv` — Eğitim verisi
- `test.csv` — Test verisi

## Gereksinimler

- Python 3.10+
- numpy, pandas, matplotlib, seaborn, scipy
- scikit-learn, xgboost, lightgbm, catboost
