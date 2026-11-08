# İst-Route Akıllı Asistan Modülü (`assistant_widget.py`) — Entegrasyon Kılavuzu

Bu paket, mevcut bir Streamlit harita uygulamasına **İst-Route Akıllı Rota ve Bakiye Asistanı**nı (`PyTorch RiskMLP` + `LangChain / Gemini` + `Floating Popover UI`) tek bir fonksiyon çağrısıyla eklemek için hazırlanmıştır.

`assistant_widget.py` içinde **`st.set_page_config` yoktur** ve **kenar çubuğuna (`st.sidebar`) müdahale etmez**; tüm `st.session_state` ve widget anahtarları `key_prefix` ile izole edilmiştir.

---

## 1. Teslim Paketi Dosya Listesi

Harita uygulamasının çalışma dizinine şu dosyaları kopyalayın:

1. `assistant_widget.py` — Yeniden kullanılabilir asistan ve PyTorch risk modeli modülü
2. `musteri_profili_ortak.csv` — PyTorch `RiskMLP` modelinin ilk açılışta önbellekli (`@st.cache_resource`) eğitimi için veri seti
3. `demo_yanitlari.json` — Kota aşımı (`429`) durumunda son çare hazır demo yanıtları
4. `requirements.txt` — Gerekli Python paketleri
5. `INTEGRATION.md` — Bu entegrasyon kılavuzu

---

## 2. 3 Satırlık Kullanım Örneği

Ana harita uygulamanızın (`map_app.py` vb.) en altına şu 3 satırı eklemeniz yeterlidir:

```python
from assistant_widget import render_assistant

aktif_rota = {"from": "Kadıköy", "to": "Levent", "summary": "Beşiktaş Vapuru -> Kabataş F1 -> Taksim M2 Metro -> Levent", "duration_min": 42}
render_assistant(route_context=aktif_rota, key_prefix="harita_asistan", show_profile_controls=True)
```

- `profile=None` bırakılırsa veri setinin medyan değerleriyle varsayılan yolcu profili otomatik oluşturulur.
- `show_profile_controls=True` verilirse kenar çubuğu kullanılmadan, sağ alttaki sohbet penceresinin (`st.popover`) içindeki açılır kutuda (`👤 Yolcu Profili Ayarları`) profil kontrolleri gösterilir.
- `route_context=None` bırakılırsa asistan soru/yanıt metninden anahtar kelimeyle rota yakalar ve `st.session_state["aktif_rota"]` değerini günceller.

---

## 3. `route_context` Sözlüğü Anahtar Yapısı

Harita modülünüzden asistana seçili güzergâhı aktarırken aşağıdaki anahtarları içeren bir `dict` gönderebilirsiniz (tüm alanlar isteğe bağlıdır):

| Anahtar | Tip | Açıklama / Örnek |
| :--- | :--- | :--- |
| `"from"` | `str` | Başlangıç noktası (`"Kadıköy İskelesi"`) |
| `"to"` | `str` | Varış noktası (`"Levent Metro Durağı"`) |
| `"summary"` | `str` | Kullanılacak araçlar ve güzergâh özeti (`"Kadıköy-Beşiktaş Vapuru -> F1 -> M2 Levent"`) |
| `"duration_min"` | `int \| float` | Tahmini yolculuk süresi (`42`) |
| `"transfers"` | `list[str] \| str` | Aktarma durakları (`["Beşiktaş İskelesi", "Taksim M2"]`) |
| `"crowd_level"` | `str` | Hat yoğunluk bilgisi (`"Düşük (Mavi Rota)"`) |
| `"alternatives"` | `str` | Alternatif Mavi Rota ve yol üstü Beltur / Kent Lokantası bilgisi |

> `route_context` verildiğinde sistem promptuna otomatik olarak `[Harita Rotası]` bölümü ve *"Rota bilgisi verilmişse yalnızca bu bilgiye dayanarak anlat; verilmeyen hat veya durak uydurma."* kuralı eklenir.

---

## 4. Gereken Paketler (`requirements.txt`)

```bash
pip install -r requirements.txt
```

Gerekli paketler:
- `streamlit>=1.32`
- `torch>=2.0`
- `pandas`
- `numpy`
- `scikit-learn`
- `langchain-google-genai>=2.0`
- `langchain-core`

---

## 5. API Anahtarının Tanımlanması

`assistant_widget.py` anahtarı aşağıdaki sırayla arar (hiçbir kaynak dosyada gerçek anahtar yazılı değildir):

1. **Ortam Değişkeni (Önerilen):**
   ```bash
   export GOOGLE_API_KEY="BURAYA_API_ANAHTARI_GELECEK"
   streamlit run demo_host_app.py
   ```
2. **Streamlit Secrets (`.streamlit/secrets.toml`):**
   ```toml
   GOOGLE_API_KEY = "BURAYA_API_ANAHTARI_GELECEK"
   ```
3. **Geliştirme Sabiti (`assistant_widget.py` içindeki `LOCAL_API_KEY`):**
   - Geliştirme sırasında `assistant_widget.py` başındaki `LOCAL_API_KEY = ""` satırına geçici olarak yapıştırılabilir.
   - **Önemli:** GitHub'a yüklemeden önce `LOCAL_API_KEY = ""` olarak boşaltın ve `python anahtar_kontrol.py` ile kontrol edin.

---

## 6. Hata Durumlarında Davranış ve Çözümler

- **API Anahtarı Eksikse:** Uygulama çökmez; sohbet penceresinde tek bir uyarı (`"API anahtarı bulunamadı. LOCAL_API_KEY sabitine veya GOOGLE_API_KEY ortam değişkenine anahtarı ekleyin."`) gösterilir.
- **Günlük Kota Dolarsa (`429 RESOURCE_EXHAUSTED`) veya Sunucu Yoğunsa (`503 UNAVAILABLE`):**
  1. Ekrana uzun ham hata basılmaz; yalnızca kısa bilgi mesajı gösterilir ve ayrıntı terminale (`stderr`) loglanır.
  2. Varsayılan yedek model (`gemini-3.5-flash` veya `GEMINI_FALLBACK_MODELS` ortam değişkeninde virgülle tanımlanan modeller) otomatik olarak denenir.
  3. Aynı `(soru, risk_seviyesi, yas_grubu, rota)` kombinasyonu daha önce sorulduysa LLM çağrılmaz, yanıt anında önbellekten (`⚡ önbellekten`) döner.
  4. Tüm modeller kota hatası verirse `demo_yanitlari.json` içindeki doğrulanmış hazır yanıtlar kullanılır.
