# Datathon Competition Solution

Bu proje, datathon yarışması için hazırlanmış bir çözümdür. Her `user_session` için `session_value` tahmin etmeyi amaçlar.

## Dosya Yapısı

- `train.csv` - Eğitim verisi (event_time, event_type, product_id, category_id, user_id, user_session, session_value)
- `test.csv` - Test verisi (event_time, event_type, product_id, category_id, user_id, user_session)
- `sample_submission.csv` - Örnek submission formatı
- `datathon_solution.py` - Ana çözüm script'i
- `requirements.txt` - Gerekli Python paketleri

## Kurulum

1. Python 3.8+ yüklü olduğundan emin olun
2. Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

## Kullanım

Çözümü çalıştırmak için:

```bash
python datathon_solution.py
```

## Çözüm Detayları

### Feature Engineering
- **event_count**: Her session'daki toplam event sayısı
- **event_type_count**: Her session'daki farklı event türü sayısı
- **product_count**: Her session'da görülen farklı ürün sayısı
- **category_count**: Her session'da görülen farklı kategori sayısı

### Model
- **Random Forest Regressor** kullanılıyor
- 100 ağaç, random_state=42
- Validation için %80 train, %20 test split

### Çıktı
- `submission.csv` dosyası oluşturulur
- Her user_session için tahmin edilen session_value

## Performans Metrikleri

Script çalıştırıldığında validation set üzerinde:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error) 
- RMSE (Root Mean Squared Error)
- Feature importance sıralaması

gösterilir.

## Notlar

- Tüm session'lar için tahmin yapılır
- Negatif değerler 0'a yuvarlanabilir (gerekirse)
- Model performansını artırmak için ek feature'lar eklenebilir
