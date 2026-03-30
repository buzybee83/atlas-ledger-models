# Atlas Ledger Models

On-device AI models for Atlas Ledger.

| Model | Version | Size | Format | Status |
|-------|---------|------|--------|--------|
| NL-to-SQL (Qwen2.5-1.5B) | v3.0.0 | ~940MB | GGUF Q4_K_M | Released |
| Receipt NER (DistilBERT INT8) | — | ~65MB | ONNX | Needs training |

---

## Receipt NER

Extracts structured fields from receipt OCR text using a fine-tuned DistilBERT token classifier.

### What it does

Given whitespace-split OCR text from a receipt, predicts a BIO tag for each token:

| Tag | Field | Example |
|-----|-------|---------|
| B-MER / I-MER | Merchant name | `STARBUCKS COFFEE` |
| B-TOT / I-TOT | Grand total | `15.60` |
| B-TAX / I-TAX | Tax amount | `0.87` |
| B-TIP / I-TIP | Tip amount | `2.00` |
| B-DATE | Date | `12/05/2025` |
| O | Other | — |

### Architecture

- **Base:** `distilbert-base-uncased` (66M params)
- **Head:** Linear token classification (10 labels)
- **Quantization:** INT8 post-training via `optimum` + ONNX Runtime
- **Inference:** `onnxruntime-react-native` v1.24.3

### Training

Open [`receipt-ner/train_receipt_ner.ipynb`](receipt-ner/train_receipt_ner.ipynb) in Google Colab.

**Requirements:** T4 GPU (free tier), ~4 hours

Steps:
1. `Runtime → Change runtime type → T4 GPU`
2. Run all cells — downloads CORD v2 + SROIE from Hugging Face automatically
3. Final cell downloads 3 release files to your machine

### Acceptance thresholds (F1)

| Field | Minimum |
|-------|---------|
| TOT (total) | 0.92 |
| DATE | 0.88 |
| MER (merchant) | 0.85 |
| TAX | 0.75 |
| TIP | 0.70 |

The notebook checks these automatically and warns before export if any threshold is missed.

### Releasing

```bash
gh release create receipt-ner-v1.0 \
  receipt-classifier.onnx vocab.txt label_map.json \
  --repo buzybee83/atlas-ledger-models \
  --title "Receipt NER v1.0" \
  --notes "DistilBERT INT8 ONNX — BIO token classifier for receipt field extraction."
```

### Wiring into the app

In `src/services/ReceiptONNXService.ts`:

```typescript
const RECEIPT_MODEL_DOWNLOAD_URL =
  'https://github.com/buzybee83/atlas-ledger-models/releases/download/receipt-ner-v1.0/receipt-classifier.onnx';
```

`vocab.txt` and `label_map.json` use the same base URL (replace filename). The app downloads and caches all three on first use.

### Files in each release

| File | Description |
|------|-------------|
| `receipt-classifier.onnx` | Quantized INT8 ONNX model |
| `vocab.txt` | DistilBERT WordPiece vocabulary |
| `label_map.json` | `{"0": "O", "1": "B-MER", ...}` |

---

## Voice Intent Classifier

Classifies voice and text utterances into 10 intent categories for hands-free expense tracking.

### Intent classes

| ID | Intent | Example |
|----|--------|---------|
| 0 | ADD_EXPENSE | "add $45 at Starbucks" |
| 1 | QUERY_SPENDING | "how much did I spend this month" |
| 2 | QUERY_BUDGET | "am I on track with my budget" |
| 3 | QUERY_CATEGORY | "how much on food this month" |
| 4 | QUERY_MERCHANT | "how much at Costco" |
| 5 | QUERY_DATE_RANGE | "expenses last quarter" |
| 6 | SET_BUDGET | "set food budget to 500" |
| 7 | LOG_MILEAGE | "log 45 miles" |
| 8 | EXPORT_DATA | "export my expenses" |
| 9 | OTHER | "never mind" |

### Architecture

- **Base:** `distilbert-base-uncased` (66M params)
- **Head:** Linear sequence classification (10 classes)
- **Quantization:** INT8 post-training via ONNX Runtime
- **Inference:** `onnxruntime-react-native` v1.24.3

### Training data

2,000 synthetic utterances (200 per class) generated with template + randomisation. No external dataset required — runs entirely in Colab free tier.

### Acceptance thresholds (F1)

| Intent | Minimum |
|--------|---------|
| ADD_EXPENSE | 0.90 |
| QUERY_* (5 classes) | 0.85 each |
| SET_BUDGET, LOG_MILEAGE, EXPORT_DATA | 0.80 each |
| OTHER | 0.75 |

### Training

Open `voice-intent/train_voice_intent.ipynb` in Google Colab. No GPU required (CPU is fine for this dataset size — ~15 minutes).

### Creating a release

```bash
gh release create voice-intent-v1.0 \
  release/voice-intent-classifier.onnx \
  release/vocab.txt \
  release/intent_map.json \
  --repo buzybee83/atlas-ledger-models \
  --title "Voice Intent v1.0" \
  --notes "DistilBERT INT8 ONNX. 10-class intent classifier."
```

### Wiring into the app

In `src/services/VoiceIntentService.ts`:

```typescript
const VOICE_INTENT_MODEL_DOWNLOAD_URL =
  'https://github.com/buzybee83/atlas-ledger-models/releases/download/voice-intent-v1.0/voice-intent-classifier.onnx';
```
