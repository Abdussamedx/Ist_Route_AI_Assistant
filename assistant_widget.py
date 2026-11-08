"""
İst-Route | Yeniden Kullanılabilir Akıllı Rota ve Bakiye Asistanı Modülü (Widget)
Ana harita uygulamasına tek fonksiyon çağrısıyla (`render_assistant`) entegre edilebilir.
"""

from __future__ import annotations

import os

os.environ["PYTHONIOENCODING"] = "utf-8"
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ---------------------------------------------------------------------------
# Sabitler ve Güvenli API Anahtarı Okuma
# ---------------------------------------------------------------------------
LOCAL_API_KEY = ""  # geliştirme sırasında anahtarı buraya yapıştır; GitHub'a yüklemeden önce BOŞALT
DEBUG_INFO = True


def _is_valid_api_key(val: str) -> bool:
    cleaned = val.strip()
    return bool(cleaned) and not cleaned.upper().startswith("BURAYA")


def get_api_key() -> str:
    # a. os.environ["GOOGLE_API_KEY"]
    env_key = os.environ.get("GOOGLE_API_KEY", "")
    if isinstance(env_key, str) and _is_valid_api_key(env_key):
        return env_key.strip()

    # b. st.secrets.get("GOOGLE_API_KEY")
    try:
        sec_key = st.secrets.get("GOOGLE_API_KEY", "")
        if sec_key is not None and _is_valid_api_key(str(sec_key)):
            return str(sec_key).strip()
    except Exception:
        pass

    # c. Dosyanın en üstünde tanımlı LOCAL_API_KEY sabiti
    if isinstance(LOCAL_API_KEY, str) and _is_valid_api_key(LOCAL_API_KEY):
        return LOCAL_API_KEY.strip()

    return ""


GOOGLE_API_KEY = get_api_key()
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
DEBUG_TIMING = True

CSV_NAME = "musteri_profili_ortak.csv"
CSV_CANDIDATES = [
    Path(__file__).resolve().parent / CSV_NAME,
    Path.cwd() / CSV_NAME,
    Path.home() / "Desktop" / CSV_NAME,
]
CAT_COLS = ["yas_grubu", "h_en_sik_mcc"]
NUM_COLS = ["h_ort_harcama", "h_aktif_gun", "m_giris_sayisi"]
FEATURES = ["yas_grubu", "h_ort_harcama", "h_aktif_gun", "h_en_sik_mcc", "m_giris_sayisi"]
TARGET = "h_basarisiz"
EPOCHS = 50
RISK_THRESHOLD = 0.5
SEED = 42

SYSTEM_PROMPT = (
    "Sen İst-Route uygulamasının samimi ulaşım ve bakiye asistanısın. Kullanıcı bir yerden bir yere "
    "nasıl gideceğini sorarsa madde işareti ve numaralı liste kullanmadan, akıcı Türkçeyle kısa bir "
    "paragraf yaz: hangi ulaşım araçlarını sırayla kullanacağını anlat, gerekirse 'kısa bir yürüyüşle "
    "... iskelesine gidin, ... vapuruna binin, ...'de metroya aktarma yapın' gibi doğal bir dille "
    "nerede inip nerede aktarma yapacağını söyle. 4-5 cümleyi geçme ve yanıtı mutlaka varış noktasına "
    "kadar tamamla. Emin olmadığın hattı veya durağı uydurma; bu durumda 'bu hattı harita ekranından "
    "doğrulayın' de. Son cümlede kısaca yoğun hat yerine Mavi Rota'yı, yol üstündeki İBB tesisinde "
    "(Beltur, Kent Lokantası) indirimi ve Parapuan kazancını an. Bakiye riski yüksekse bir cümleyle "
    "bakiye yüklemeyi hatırlat."
)

ROUTE_CONTEXT_RULE = (
    "Rota bilgisi verilmişse yalnızca bu bilgiye dayanarak anlat; verilmeyen hat veya durak uydurma."
)

AUTO_PROMPT = "Profilime göre bakiye riskimi kısaca değerlendir; Mavi Rota ve Beltur/Parapuan fırsatını öner. En fazla 3 cümle."

QUICK_PROMPTS = [
    "Trafiğe Girmeden Nasıl Giderim?",
    "İBB Tesislerinde İndirim Fırsatları",
]

# Harita modülü köprüsü için anahtar kelime -> rota eşleşmeleri
ROUTE_RULES: list[tuple[list[str], str]] = [
    (["levent"], "Levent - Mavi Rota (M2 / Alternatif Hat, Beltur İndirimli)"),
    (["cevizlibağ", "cevizlibag"], "Cevizlibağ - Mavi Rota (T1 / Marmaray Alternatifi, Kent Lokantası Noktası)"),
    (["kadıköy", "kadikoy"], "Kadıköy - Mavi Rota (Şehir Hatları Vapur + Beltur Sahil)"),
    (["beşiktaş", "besiktas"], "Beşiktaş - Mavi Rota (Deniz Ulaşımı + Beltur İskele)"),
    (["taksim"], "Taksim - Mavi Rota (F1 / M2 Alternatif Güzergâhı)"),
    (["üsküdar", "uskudar"], "Üsküdar - Mavi Rota (Marmaray / Vapur + Beltur)"),
    (["mecidiyeköy", "mecidiyekoy", "şişli", "sisli"], "Mecidiyeköy - Mavi Rota (M7 / M2 Alternatif Hat)"),
    (["mavi rota", "beltur", "kent lokantası", "kent lokantasi"], "Mavi Rota (Alternatif Boş Hat & İBB Beltur / Kent Lokantası Güzergâhı)"),
]


def detect_active_route(*texts: str) -> str | None:
    """Soru veya asistan yanıtı içinde geçen lokasyon/anahtar kelimelere göre aktif rotayı belirler."""
    combined = " ".join(t for t in texts if t).lower()
    for keywords, route_label in ROUTE_RULES:
        if any(k in combined for k in keywords):
            return route_label
    return None


# ---------------------------------------------------------------------------
# Veri hazırlama + PyTorch MLP (Önbellekli)
# ---------------------------------------------------------------------------
class RiskMLP(nn.Module):
    """5 -> 16 -> 8 -> 1 basit ikili sınıflandırıcı (logit çıktısı)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def find_csv() -> Path | None:
    for p in CSV_CANDIDATES:
        try:
            if p.is_file():
                return p
        except OSError:
            continue
    return None


def read_csv_utf8(source) -> pd.DataFrame:
    """Türkçe karakterleri hatasız okumak için sırasıyla utf-8, utf-8-sig ve latin5 dener."""
    for enc in ("utf-8", "utf-8-sig", "latin5"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(source, encoding="utf-8", errors="replace")


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in CAT_COLS:
        df[c] = df[c].fillna("Bilinmiyor").astype(str).str.strip()
    for c in NUM_COLS + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


@st.cache_data(show_spinner=False)
def load_data(csv_bytes: bytes | None, csv_path: str | None) -> pd.DataFrame:
    """musteri_profili_ortak.csv sadece bir kez okunur ve önbelleğe alınır."""
    if csv_bytes is not None:
        import io

        raw = read_csv_utf8(io.BytesIO(csv_bytes))
    else:
        raw = read_csv_utf8(csv_path)

    missing = [c for c in FEATURES + [TARGET] if c not in raw.columns]
    if missing:
        raise ValueError(f"CSV'de eksik kolonlar: {missing}")

    return clean_frame(raw[FEATURES + [TARGET]])


@st.cache_resource(show_spinner=False)
def train_model(csv_bytes: bytes | None = None, csv_path: str | None = None):
    """PyTorch modelini yalnızca ilk açılışta bir kez eğitir/yükler ve bellekte tutar."""
    if csv_bytes is None and csv_path is None:
        found = find_csv()
        if found is None:
            raise FileNotFoundError(f"{CSV_NAME} bulunamadı.")
        csv_path = str(found)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df = load_data(csv_bytes, csv_path)

    encoders: dict[str, LabelEncoder] = {}
    encoded = df[FEATURES].copy()
    for c in CAT_COLS:
        le = LabelEncoder()
        encoded[c] = le.fit_transform(encoded[c])
        encoders[c] = le

    encoded["h_ort_harcama"] = np.log1p(encoded["h_ort_harcama"].clip(lower=0))

    X = encoded[FEATURES].values.astype(np.float32)
    y = (df[TARGET].values > 0).astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y if len(np.unique(y)) > 1 else None
    )

    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = np.clip(scaler.transform(X_te), 0.0, 1.0).astype(np.float32)

    model = RiskMLP(in_dim=len(FEATURES))
    pos_weight = torch.tensor([(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    xt = torch.from_numpy(X_tr)
    yt = torch.from_numpy(y_tr).unsqueeze(1)

    model.train()
    losses: list[float] = []
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = criterion(model(xt), yt)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(X_te))).squeeze(1).numpy()

    preds = (probs >= RISK_THRESHOLD).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_te, preds)),
        "roc_auc": float(roc_auc_score(y_te, probs)) if len(np.unique(y_te)) > 1 else 0.5,
        "pos_rate": float(y.mean()),
        "n_rows": int(len(df)),
        "risk_rate": float(y.mean()),
        "losses": losses,
    }
    stats = {
        "harcama_max": float(np.percentile(df["h_ort_harcama"], 99)),
        "harcama_median": float(df["h_ort_harcama"].median()),
        "giris_max": int(max(df["m_giris_sayisi"].max(), 1)),
    }
    return model, scaler, encoders, metrics, stats


def get_default_profile(encoders: dict, stats: dict) -> dict:
    """profile=None verildiğinde veri setinin medyanlarıyla varsayılan bir profil oluşturur."""
    yas_opts = [c for c in encoders["yas_grubu"].classes_ if c != "Bilinmiyor"] or list(
        encoders["yas_grubu"].classes_
    )
    mcc_opts = list(encoders["h_en_sik_mcc"].classes_)
    default_mcc = next((m for m in mcc_opts if "TAŞIMACILIK" in m), mcc_opts[0])
    return {
        "yas_grubu": yas_opts[0],
        "h_ort_harcama": float(round(stats["harcama_median"])),
        "h_aktif_gun": 5,
        "h_en_sik_mcc": default_mcc,
        "m_giris_sayisi": 1,
    }


@st.cache_data(show_spinner=False)
def predict_risk(
    _model,
    _scaler,
    _encoders,
    yas_grubu: str,
    h_ort_harcama: float,
    h_aktif_gun: int,
    h_en_sik_mcc: str,
    m_giris_sayisi: int,
) -> float:
    """Profil bazlı risk skorunu önbelleğe alarak her etkileşimde tekrar hesaplanmasını engeller."""
    profile_map = {
        "yas_grubu": yas_grubu,
        "h_ort_harcama": h_ort_harcama,
        "h_aktif_gun": h_aktif_gun,
        "h_en_sik_mcc": h_en_sik_mcc,
        "m_giris_sayisi": m_giris_sayisi,
    }
    row = []
    for c in FEATURES:
        v = profile_map[c]
        if c in CAT_COLS:
            le = _encoders[c]
            v = v if v in le.classes_ else le.classes_[0]
            row.append(float(le.transform([v])[0]))
        elif c == "h_ort_harcama":
            row.append(float(np.log1p(max(v, 0.0))))
        else:
            row.append(float(v))
    x = _scaler.transform(np.array([row], dtype=np.float32)).astype(np.float32)
    x = np.clip(x, 0.0, 1.0)
    with torch.no_grad():
        return float(torch.sigmoid(_model(torch.from_numpy(x))).item())


# ---------------------------------------------------------------------------
# LangChain + Gemini (Önbellekli İstemci + Kota/Yedek + Süre Ölçümlü Streaming)
# ---------------------------------------------------------------------------
def build_context(profile: dict, risk: float) -> str:
    level = "YÜKSEK" if risk >= RISK_THRESHOLD else "DÜŞÜK"
    ctx = (
        f"[ML Modeli Çıktısı] Bakiye Yetersizliği Risk Skoru: %{risk * 100:.1f} (Seviye: {level}).\n"
        f"[Kullanıcı Profili] Yaş grubu: {profile['yas_grubu']}, "
        f"aylık ortalama harcama/bakiye: {profile['h_ort_harcama']:.0f} TL, "
        f"aktif gün: {profile['h_aktif_gun']}, en sık harcama kategorisi: {profile['h_en_sik_mcc']}, "
        f"mobil uygulama giriş sayısı: {profile['m_giris_sayisi']}."
    )
    if level == "YÜKSEK":
        ctx += (
            "\n[Talimat] Risk yüksek: Kullanıcıyı turnikede bakiye yetersizliği yaşayabileceği "
            "konusunda nazikçe ve proaktif olarak uyar, İstanbulkart'a bakiye yüklemesini öner, "
            "Mavi Rota'yı ve yol üstündeki Beltur indirimi ile Parapuan kazancını vurgula."
        )
    else:
        ctx += "\n[Talimat] Risk düşük: Kullanıcıya yine de Mavi Rota ve Beltur/Parapuan fırsatlarını hatırlat."
    return ctx


def format_route_context(route_context: dict) -> str:
    """Harita uygulamasından gelen route_context sözlüğünü [Harita Rotası] metnine dönüştürür."""
    key_labels = {
        "from": "Başlangıç",
        "to": "Varış",
        "summary": "Rota Özeti",
        "duration_min": "Tahmini Süre (dk)",
        "transfers": "Aktarmalar",
        "crowd_level": "Yoğunluk Seviyesi",
        "alternatives": "Mavi Rota / Alternatifler",
    }
    lines = ["[Harita Rotası]"]
    for k, label in key_labels.items():
        if k in route_context and route_context[k] not in (None, ""):
            val = route_context[k]
            if isinstance(val, (list, tuple)):
                val = ", ".join(str(x) for x in val)
            lines.append(f"- {label}: {val}")
    for k, v in route_context.items():
        if k not in key_labels and v not in (None, ""):
            lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def build_prompt_text(
    profile: dict,
    risk: float,
    user_input: str,
    route_context: dict | None = None,
) -> str:
    """Sistem promptu, (varsa) harita rotası, risk bağlamı ve kullanıcı sorusunu birleştirir."""
    sys_prompt = SYSTEM_PROMPT
    if route_context:
        sys_prompt = f"{SYSTEM_PROMPT} {ROUTE_CONTEXT_RULE}"

    context_info = build_context(profile, risk)
    parts = [sys_prompt]
    if route_context:
        parts.append(format_route_context(route_context))
    parts.append(context_info)
    parts.append(f"Kullanıcı: {user_input}")
    return "\n\n".join(parts)


QUOTA_SHORT_ERROR = "Günlük ücretsiz kota doldu. Yedek modele geçiliyor / hazır yanıt gösteriliyor."
DEMO_UNAVAILABLE_MSG = "Asistan şu an kota nedeniyle kullanılamıyor. Hazır örnek sorulardan birini deneyin."
DEMO_JSON_PATH = Path(__file__).resolve().parent / "demo_yanitlari.json"


def get_fallback_models() -> list[str]:
    raw = os.environ.get("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash")
    return [m.strip() for m in raw.split(",") if m.strip()]


def _mask_key(text: object) -> str:
    import re

    pat = r"(?:AI" + r"za|AQ\.)[0-9A-Za-z_-]{10,}"
    return re.sub(pat, "[MASKED_KEY]", str(text))


def is_quota_error(exc: Exception) -> bool:
    combined = f"{exc} {repr(exc)} {exc.__cause__} {exc.__context__}"
    return "429" in combined or "RESOURCE_EXHAUSTED" in combined


def is_transient_or_quota_error(exc: Exception) -> bool:
    combined = f"{type(exc).__name__} {exc} {repr(exc)} {exc.__cause__} {exc.__context__}"
    return (
        is_quota_error(exc)
        or "503" in combined
        or "UNAVAILABLE" in combined
        or "ReadTimeout" in combined
        or "timed out" in combined.lower()
    )


def log_api_error_safely(exc: Exception) -> None:
    import traceback

    tb_str = _mask_key(traceback.format_exc())
    details = (
        f"[LLM API Hatası Ayrıntısı]\n"
        f"  type(e).__name__: {type(exc).__name__}\n"
        f"  str(e)          : {_mask_key(str(exc))}\n"
        f"  repr(e)         : {_mask_key(repr(exc))}\n"
        f"  e.__cause__     : {_mask_key(repr(exc.__cause__))}\n"
        f"  e.__context__   : {_mask_key(repr(exc.__context__))}\n"
        f"  traceback       :\n{tb_str}"
    )
    print(details, file=sys.stderr, flush=True)


def format_ui_api_error(exc: Exception) -> str:
    raw_msg = _mask_key(str(exc)).strip()
    detail = raw_msg[:200] if raw_msg else "(mesaj boş, ayrıntı terminalde)"
    return f"API Hatası: {type(exc).__name__} — {detail}"


def normalize_question(text: str) -> str:
    return " ".join(text.strip().lower().split())


def get_demo_answer(user_input: str) -> str | None:
    if not DEMO_JSON_PATH.is_file():
        return None
    try:
        import json

        data = json.loads(DEMO_JSON_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    norm_q = normalize_question(user_input)
    norm_q_tr = norm_q.replace("i\u0307", "i")
    for k, v in data.items():
        if not isinstance(v, str):
            continue
        val = v.strip()
        if not val or val.upper().startswith("BURAYA"):
            continue
        k_norm = normalize_question(str(k))
        if k_norm == norm_q or k_norm.replace("i\u0307", "i") == norm_q_tr:
            return val
    return None


@st.cache_resource(show_spinner=False)
def get_cached_llm(api_key: str, model_name: str = MODEL_NAME):
    """ChatGoogleGenerativeAI istemcisini bir kez oluşturup bellekte tutar."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    stripped_key = api_key.strip()
    clean_key = "".join(ch for ch in stripped_key if ord(ch) < 128)
    if len(clean_key) != len(stripped_key):
        print("anahtarda ASCII dışı karakter vardı", file=sys.stderr, flush=True)
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=clean_key,
        temperature=0.4,
        max_output_tokens=1024,
        timeout=40,
        max_retries=1,
        thinking_budget=0,
        # transport="rest",
    )


def extract_chunk_text(chunk) -> str:
    """chunk.content string veya list (content blocks) gelebileceği için her durumda string'e çevirir."""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "".join(parts)
    return str(content) if content else ""


def stream_answer(llm, prompt_text: str) -> str:
    t0 = time.perf_counter()
    first = {"t": None}

    def gen():
        for chunk in llm.stream(prompt_text):
            piece = extract_chunk_text(chunk)
            if piece:
                if first["t"] is None:
                    first["t"] = time.perf_counter() - t0
                yield piece

    out = st.write_stream(gen())
    if DEBUG_TIMING and first["t"] is not None:
        st.caption(f"⏱ ilk token {first['t']:.1f} sn · toplam {time.perf_counter() - t0:.1f} sn")
    return out if isinstance(out, str) else "".join(str(x) for x in out)


def stream_synthetic_answer(text: str) -> str:
    """Hazır demo yanıtlarını st.write_stream ile kısa bir yapay akıtmayla gösterir."""
    def gen():
        words = text.split(" ")
        for idx, word in enumerate(words):
            yield word + (" " if idx < len(words) - 1 else "")
            time.sleep(0.01)

    out = st.write_stream(gen())
    return out if isinstance(out, str) else "".join(str(x) for x in out)


# ---------------------------------------------------------------------------
# Yüzen (Floating) Popover CSS
# ---------------------------------------------------------------------------
FLOATING_UX_CSS = """
<style>
/* Sağ alt köşe yüzen konuşma balonu (Tooltip) */
.floating-tooltip {
    position: fixed;
    bottom: 84px;
    right: 28px;
    z-index: 999998;
    background: #ffffff;
    color: #0b3d91;
    padding: 9px 15px;
    border-radius: 16px 16px 4px 16px;
    box-shadow: 0 8px 24px rgba(11, 61, 145, 0.22);
    font-size: 0.85rem;
    font-weight: 600;
    border: 1.5px solid #00a3e0;
    pointer-events: none;
    animation: floatPulse 3s ease-in-out infinite;
}
@keyframes floatPulse {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-4px); }
}

/* Sağ alt köşe yüzen st.popover butonu */
div[data-testid="stPopover"] {
    position: fixed !important;
    bottom: 24px !important;
    right: 28px !important;
    z-index: 999999 !important;
    width: auto !important;
}
div[data-testid="stPopover"] > div > button {
    border-radius: 999px !important;
    height: 52px !important;
    padding: 0 22px !important;
    background: linear-gradient(135deg, #0b3d91 0%, #00a3e0 100%) !important;
    color: #ffffff !important;
    border: 2px solid #ffffff !important;
    box-shadow: 0 10px 28px rgba(11, 61, 145, 0.38) !important;
    font-weight: 700 !important;
    font-size: 0.96rem !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
div[data-testid="stPopover"] > div > button:hover {
    transform: scale(1.04) !important;
    box-shadow: 0 14px 34px rgba(0, 163, 224, 0.52) !important;
}

/* Açılan Pop-up sohbet penceresi boyutları */
div[data-testid="stPopoverBody"] {
    width: 410px !important;
    max-width: 92vw !important;
    border-radius: 18px !important;
    box-shadow: 0 20px 48px rgba(15, 23, 42, 0.28) !important;
    border: 1px solid #e2e8f0 !important;
}

/* Sohbet giriş kutusundaki 'Press Enter to submit form' yazısını gizle */
div[data-testid="InputInstructions"] { display: none !important; }
</style>
"""


# ---------------------------------------------------------------------------
# Tek Giriş Fonksiyonu: render_assistant
# ---------------------------------------------------------------------------
def render_assistant(
    route_context: dict | None = None,
    profile: dict | None = None,
    key_prefix: str = "ist_asistan",
    show_profile_controls: bool = False,
) -> dict:
    """
    İst-Route yüzen sohbet asistanını mevcut Streamlit sayfasına ekler.
    Sayfa ayarı (set_page_config) veya zorunlu kenar çubuğu (sidebar) içermez.
    """
    st.markdown(FLOATING_UX_CSS, unsafe_allow_html=True)

    model, scaler, encoders, metrics, stats = train_model()
    api_key = get_api_key()

    # Session state anahtarları (key_prefix ile izole)
    k_messages = f"{key_prefix}_messages"
    k_error = f"{key_prefix}_last_api_error"
    k_model = f"{key_prefix}_active_model"
    k_count = f"{key_prefix}_llm_request_count"
    k_cache = f"{key_prefix}_response_cache"
    k_llm = f"{key_prefix}_llm"

    if k_model not in st.session_state:
        st.session_state[k_model] = MODEL_NAME
    if k_count not in st.session_state:
        st.session_state[k_count] = 0
    if k_cache not in st.session_state:
        st.session_state[k_cache] = {}
    if k_messages not in st.session_state:
        st.session_state[k_messages] = [
            {
                "role": "assistant",
                "content": (
                    "Merhaba! 👋 Ben **İst-Route** akıllı ulaşım ve bakiye asistanınızım. "
                    "Size Mavi Rota, bakiye durumunuz veya İBB tesislerindeki indirimler "
                    "hakkında nasıl yardımcı olabilirim?"
                ),
            }
        ]
    if k_error not in st.session_state:
        st.session_state[k_error] = None

    st.session_state[k_llm] = get_cached_llm(api_key, st.session_state[k_model]) if api_key else None

    if profile is None:
        active_profile = get_default_profile(encoders, stats)
    else:
        active_profile = dict(profile)

    # Sağ alt köşeye CSS ile sabitlenmiş st.popover bileşeni
    with st.popover("💬 İst-Route Asistanı"):
        st.markdown("##### 🤖 İst-Route Akıllı Asistan")

        if show_profile_controls:
            with st.expander("👤 Yolcu Profili Ayarları", expanded=False):
                yas_opts = [c for c in encoders["yas_grubu"].classes_ if c != "Bilinmiyor"] or list(
                    encoders["yas_grubu"].classes_
                )
                mcc_opts = list(encoders["h_en_sik_mcc"].classes_)
                default_yas_idx = (
                    yas_opts.index(active_profile["yas_grubu"])
                    if active_profile.get("yas_grubu") in yas_opts
                    else 0
                )
                default_mcc_idx = (
                    mcc_opts.index(active_profile["h_en_sik_mcc"])
                    if active_profile.get("h_en_sik_mcc") in mcc_opts
                    else next((i for i, m in enumerate(mcc_opts) if "TAŞIMACILIK" in m), 0)
                )
                yas_val = st.selectbox(
                    "Yaş Grubu",
                    yas_opts,
                    index=default_yas_idx,
                    key=f"{key_prefix}_ctrl_yas",
                )
                harcama_val = st.slider(
                    "Ortalama Bakiye / Harcama (TL)",
                    0.0,
                    float(round(stats["harcama_max"], -1) or 5000.0),
                    float(active_profile.get("h_ort_harcama", round(stats["harcama_median"]))),
                    step=10.0,
                    key=f"{key_prefix}_ctrl_harcama",
                )
                aktif_val = st.slider(
                    "Aktif Gün (aylık)",
                    0,
                    30,
                    int(active_profile.get("h_aktif_gun", 5)),
                    key=f"{key_prefix}_ctrl_aktif",
                )
                mcc_val = st.selectbox(
                    "En Sık Harcama Kategorisi",
                    mcc_opts,
                    index=default_mcc_idx,
                    key=f"{key_prefix}_ctrl_mcc",
                )
                giris_val = st.slider(
                    "Mobil Uygulama Giriş Sayısı",
                    0,
                    int(min(stats["giris_max"], 50)),
                    int(active_profile.get("m_giris_sayisi", 1)),
                    key=f"{key_prefix}_ctrl_giris",
                )
                active_profile = {
                    "yas_grubu": yas_val,
                    "h_ort_harcama": float(harcama_val),
                    "h_aktif_gun": int(aktif_val),
                    "h_en_sik_mcc": mcc_val,
                    "m_giris_sayisi": int(giris_val),
                }

        risk = predict_risk(
            model,
            scaler,
            encoders,
            active_profile["yas_grubu"],
            float(active_profile["h_ort_harcama"]),
            int(active_profile["h_aktif_gun"]),
            active_profile["h_en_sik_mcc"],
            int(active_profile["m_giris_sayisi"]),
        )
        high = risk >= RISK_THRESHOLD
        harcama = float(active_profile["h_ort_harcama"])

        status_txt = f"🔴 Yüksek Risk (%{risk * 100:.0f})" if high else f"🟢 Risksiz (%{risk * 100:.0f})"
        st.caption(f"Bakiye Durumu: **{status_txt}** · Ortalama: **{harcama:.0f} TL**")

        if route_context:
            r_from = route_context.get("from", "Başlangıç")
            r_to = route_context.get("to", "Varış")
            r_sum = route_context.get("summary", "")
            st.caption(f"🗺️ Aktif Harita Rotası: **{r_from} → {r_to}**" + (f" ({r_sum})" if r_sum else ""))

        if high:
            st.warning(
                f"Bakiye riski %{risk * 100:.0f}. Yola çıkmadan İstanbulkart'a bakiye yükleyin; "
                "Mavi Rota ve Beltur/Parapuan fırsatı için aşağıdan asistana sorun."
            )

        # --- Hızlı aksiyon butonları (pencerenin üst kısmında) ---
        quick_highlight_placeholder = st.empty()
        quick_question = None
        qb1, qb2 = st.columns(2)
        if qb1.button(
            "🚦 Trafiğe Girmeden Nasıl Giderim?",
            use_container_width=True,
            key=f"{key_prefix}_q_btn_1",
        ):
            quick_question = QUICK_PROMPTS[0]
        if qb2.button(
            "🎁 İBB Tesislerinde İndirim Fırsatları",
            use_container_width=True,
            key=f"{key_prefix}_q_btn_2",
        ):
            quick_question = QUICK_PROMPTS[1]
        if st.button(
            "⚠️ Bana özel bakiye uyarısı",
            use_container_width=True,
            key=f"{key_prefix}_q_btn_3",
        ):
            quick_question = AUTO_PROMPT

        # --- Kaydırılabilir kompakt sohbet geçmişi ---
        chat_container = st.container(height=300, border=True)
        with chat_container:
            for m in st.session_state[k_messages]:
                with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🚇"):
                    st.markdown(m["content"])

        error_placeholder = st.empty()

        # --- Kompakt mesaj gönderme formu ---
        with st.form(key=f"{key_prefix}_popup_chat_form", clear_on_submit=True, border=False):
            in_col, btn_col = st.columns([4, 1])
            with in_col:
                user_text = st.text_input(
                    "Mesajınız",
                    placeholder="Rotanızı veya sorunuzu yazın...",
                    label_visibility="collapsed",
                    key=f"{key_prefix}_chat_text_input",
                )
            with btn_col:
                send_clicked = st.form_submit_button("Gönder", use_container_width=True)

        submitted_question = None
        if send_clicked and user_text.strip():
            submitted_question = user_text.strip()
        elif quick_question:
            submitted_question = quick_question

        if submitted_question:
            user_input = submitted_question
            st.session_state[k_messages].append({"role": "user", "content": user_input})
            with chat_container:
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(user_input)

            risk_level = "YÜKSEK" if high else "DÜŞÜK"
            route_sig = (
                f"{route_context.get('from', '')}->{route_context.get('to', '')}:{route_context.get('summary', '')}"
                if route_context
                else ""
            )
            cache_key = (normalize_question(user_input), risk_level, active_profile["yas_grubu"], route_sig)

            # Önbellek kontrolü
            if cache_key in st.session_state[k_cache]:
                cached_response = st.session_state[k_cache][cache_key]
                with chat_container:
                    with st.chat_message("assistant", avatar="🚇"):
                        st.markdown(cached_response)
                        st.caption("⚡ önbellekten")
                st.session_state[k_messages].append({"role": "assistant", "content": cached_response})
                if not route_context:
                    detected_route = detect_active_route(user_input, cached_response)
                    if detected_route:
                        st.session_state["aktif_rota"] = detected_route
                        st.session_state[f"{key_prefix}_aktif_rota"] = detected_route
            elif not api_key or st.session_state[k_llm] is None:
                st.session_state[k_error] = (
                    "API anahtarı bulunamadı. LOCAL_API_KEY sabitine veya "
                    "GOOGLE_API_KEY ortam değişkenine anahtarı ekleyin."
                )
                demo_ans = get_demo_answer(user_input)
                if demo_ans:
                    with chat_container:
                        with st.chat_message("assistant", avatar="🚇"):
                            full_response = stream_synthetic_answer(demo_ans)
                    st.session_state[k_messages].append({"role": "assistant", "content": full_response})
                    if not route_context:
                        detected_route = detect_active_route(user_input, full_response)
                        if detected_route:
                            st.session_state["aktif_rota"] = detected_route
                            st.session_state[f"{key_prefix}_aktif_rota"] = detected_route
            else:
                prompt_text = build_prompt_text(active_profile, risk, user_input, route_context=route_context)
                current_model = st.session_state[k_model]
                fallback_list = [m for m in get_fallback_models() if m != current_model]
                candidate_models = [current_model] + fallback_list

                full_response = None
                quota_exhausted = False

                for candidate_model in candidate_models:
                    candidate_llm = get_cached_llm(api_key, candidate_model)
                    st.session_state[k_count] += 1
                    try:
                        with chat_container:
                            with st.chat_message("assistant", avatar="🚇"):
                                full_response = stream_answer(candidate_llm, prompt_text)
                        st.session_state[k_model] = candidate_model
                        st.session_state[k_llm] = candidate_llm
                        if not quota_exhausted:
                            st.session_state[k_error] = None
                        break
                    except Exception as e:
                        log_api_error_safely(e)
                        if is_quota_error(e):
                            quota_exhausted = True
                            st.session_state[k_error] = QUOTA_SHORT_ERROR
                            continue
                        elif is_transient_or_quota_error(e):
                            st.session_state[k_error] = format_ui_api_error(e)
                            continue
                        else:
                            st.session_state[k_error] = format_ui_api_error(e)
                            break

                if full_response is not None and full_response.strip():
                    st.session_state[k_cache][cache_key] = full_response
                    st.session_state[k_messages].append({"role": "assistant", "content": full_response})
                    if not route_context:
                        detected_route = detect_active_route(user_input, full_response)
                        if detected_route:
                            st.session_state["aktif_rota"] = detected_route
                            st.session_state[f"{key_prefix}_aktif_rota"] = detected_route
                elif quota_exhausted:
                    demo_ans = get_demo_answer(user_input)
                    if demo_ans:
                        with chat_container:
                            with st.chat_message("assistant", avatar="🚇"):
                                full_response = stream_synthetic_answer(demo_ans)
                        st.session_state[k_messages].append({"role": "assistant", "content": full_response})
                        if not route_context:
                            detected_route = detect_active_route(user_input, full_response)
                            if detected_route:
                                st.session_state["aktif_rota"] = detected_route
                                st.session_state[f"{key_prefix}_aktif_rota"] = detected_route
                    else:
                        with chat_container:
                            with st.chat_message("assistant", avatar="🚇"):
                                st.markdown(DEMO_UNAVAILABLE_MSG)
                        quick_highlight_placeholder.info(
                            "👆 Hazır örnek sorulardan birini aşağıdaki hızlı aksiyon butonlarıyla deneyebilirsiniz."
                        )
                        st.session_state[k_messages].append({"role": "assistant", "content": DEMO_UNAVAILABLE_MSG})

        if st.session_state[k_error]:
            error_placeholder.error(st.session_state[k_error])

    # Yüzen butonun hemen üstündeki konuşma balonu (Tooltip)
    badge = "🔴 " if high else "🟢 "
    st.markdown(
        f'<div class="floating-tooltip">{badge}Ben İst-Route, size nasıl yardımcı olabilirim?</div>',
        unsafe_allow_html=True,
    )

    # Geriye dönük uyumluluk için ana session_state aynalaması
    st.session_state["messages"] = st.session_state[k_messages]
    st.session_state["last_api_error"] = st.session_state[k_error]
    st.session_state["active_model"] = st.session_state[k_model]
    st.session_state["llm_request_count"] = st.session_state[k_count]
    st.session_state["response_cache"] = st.session_state[k_cache]

    return {
        "profile": active_profile,
        "risk": risk,
        "high_risk": high,
        "active_model": st.session_state[k_model],
        "llm_request_count": st.session_state[k_count],
        "aktif_rota": st.session_state.get("aktif_rota"),
    }
