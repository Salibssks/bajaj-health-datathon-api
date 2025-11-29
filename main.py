import io
import os
import base64
import json
from typing import List, Dict, Any

import fitz  # PyMuPDF
import requests
from PIL import Image

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rapidfuzz import fuzz
from dotenv import load_dotenv
import openai

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment variables.")

openai.api_key = OPENAI_API_KEY

app = FastAPI(title="Bajaj Health Datathon Bill Extraction API")

# ----------------- Request / response models ----------------- #

class ExtractRequest(BaseModel):
    document: str  # public URL of PDF / image


class LineItem:
    """
    Internal container so we can deduplicate / transform easily.
    """
    def __init__(self, name: str, amount: float, rate: float, qty: float):
        self.name = (name or "").strip()
        self.amount = float(amount) if amount is not None else 0.0
        self.rate = float(rate) if rate is not None else 0.0
        self.qty = float(qty) if qty is not None else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_name": self.name,
            "item_amount": round(self.amount, 2),
            "item_rate": round(self.rate, 2),
            "item_quantity": round(self.qty, 2),
        }

# token usage tracker (simple global)
token_usage = {
    "total_tokens": 0,
    "input_tokens": 0,
    "output_tokens": 0,
}

# ----------------- Helpers: document loading ----------------- #

def download_bytes(url: str) -> bytes:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")


def document_to_images(url: str) -> List[Image.Image]:
    """
    Handle both PDFs and direct image URLs.
    - For PDFs: convert each page to a PNG image.
    - For PNG/JPEG: return single image.
    """
    data = download_bytes(url)

    # crude check by extension
    lower = url.lower()
    if lower.endswith(".pdf"):
        doc = fitz.open(stream=data, filetype="pdf")
        pages = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x for clarity
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pages.append(img)
        doc.close()
        return pages

    # treat as image
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return [img]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unsupported document format: {e}")

# ----------------- Helpers: LLM extraction ----------------- #

def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_llm_for_page(image: Image.Image, page_no: int) -> Dict[str, Any]:
    """
    Call GPT‑4o (or similar) with vision to get:
    {
      "page_type": "Bill Detail" | "Final Bill" | "Pharmacy",
      "bill_items": [
        { "item_name": str, "item_amount": float, "item_rate": float, "item_quantity": float }
      ]
    }
    """
    global token_usage

    img_b64 = image_to_base64(image)

    prompt = f"""
You are extracting structured BILL LINE ITEMS from a medical / pharmacy bill page.

Return STRICT JSON with this structure ONLY:

{{
  "page_type": "Bill Detail" | "Final Bill" | "Pharmacy",
  "bill_items": [
    {{
      "item_name": "string",
      "item_amount": number,
      "item_rate": number,
      "item_quantity": number
    }}
  ]
}}

Rules (very important):
- A LINE ITEM must have (a) description text AND (b) a quantity AND (c) a numeric amount in the row.
- Use EXACT text for item_name from the row (no paraphrasing).
- item_amount = final net amount for that row (after discounts/taxes if shown per-line).
- item_rate = unit rate/price in that row if visible, else 0.
- item_quantity = quantity / QTY / NO. / PCS / etc. in that row if visible, else 1.
- IGNORE rows that are section totals or bill summaries, such as:
  "Sub Total", "Total", "Total of ...", "Grand Total", "Bill Amount",
  "Net Amount", "Net Payable", "Balance Amt", "Service Amount",
  "Total Payable Amount", "Discount", "Deposit", "Paid Amount".
- For pages that are clearly only summary / totals and no per-row Qty,
  set "bill_items": [] and choose page_type "Final Bill".
- For pages that are mainly pharmacy / drug lists, use page_type "Pharmacy".
- Otherwise, use page_type "Bill Detail".
"""
    try:
        # Use ChatCompletion.create style compatible with more openai versions
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # or "gpt-4o" if you have access
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": f"Image (base64 PNG): {img_b64}",
                },
            ],
            temperature=0.1,
            max_tokens=1000,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    # token accounting (may not always be present)
    if hasattr(resp, "usage") and resp.usage:
        usage = resp.usage
        token_usage["total_tokens"] += usage.total_tokens
        token_usage["input_tokens"] += usage.prompt_tokens
        token_usage["output_tokens"] += usage.completion_tokens

    # NOTE: ChatCompletion.create returns dict-style objects
    content = resp.choices[0]["message"]["content"].strip()


    # try to isolate JSON
    start = content.find("{")
    end = content.rfind("}") + 1
    if start == -1 or end == 0:
        raise HTTPException(status_code=500, detail="LLM response did not contain JSON.")

    try:
        data = json.loads(content[start:end])
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM JSON: {e}")

# ----------------- Helpers: post-processing ----------------- #

def is_summary_or_total_row(name: str) -> bool:
    """
    Heuristic to drop subtotal/total-like lines even if LLM included them.
    """
    if not name:
        return False
    text = name.lower()
    keywords = [
        "sub total", "subtotal", "total amount", "total :", "grand total",
        "bill amount", "net amount", "net amt", "net payable", "balance amt",
        "service amount", "amount in words", "discount", "deposit",
        "round off", "roundoff", "to pay", "paid amount", "due amount",
        "refund amount",
    ]
    return any(k in text for k in keywords)


def deduplicate_items(items: List[LineItem]) -> List[LineItem]:
    """
    Remove obvious duplicates across pages:
    - same name (fuzzy) and same amount -> merge quantities.
    """
    unique: List[LineItem] = []
    seen: Dict[str, int] = {}

    for item in items:
        key_base = item.name.lower().strip()
        key = f"{key_base[:50]}|{round(item.amount, 2)}"
        if key not in seen:
            seen[key] = len(unique)
            unique.append(item)
        else:
            idx = seen[key]
            # if the names are very similar, merge
            if fuzz.ratio(item.name, unique[idx].name) > 90:
                unique[idx].qty += item.qty
            else:
                unique.append(item)

    return unique

# ----------------- API endpoint ----------------- #

@app.post("/extract-bill-data")
def extract_bill_data(req: ExtractRequest):
    global token_usage
    token_usage = {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}

    # 1) Load document, convert to images
    images = document_to_images(req.document)
    if not images:
        raise HTTPException(status_code=400, detail="No pages found in document.")

    all_items: List[LineItem] = []
    pagewise_output: List[Dict[str, Any]] = []

    # 2) Process each page with LLM
    for idx, img in enumerate(images, start=1):
        page_data = call_llm_for_page(img, idx)

        page_type = page_data.get("page_type", "Bill Detail") or "Bill Detail"
        raw_items = page_data.get("bill_items", []) or []

        page_items_out: List[Dict[str, Any]] = []

        for row in raw_items:
            try:
                name = row.get("item_name", "").strip()
                amount = float(row.get("item_amount", 0) or 0)
                rate = float(row.get("item_rate", 0) or 0)
                qty = float(row.get("item_quantity", 0) or 0)

                # drop zero/negative and summary-like rows
                if amount <= 0:
                    continue
                if is_summary_or_total_row(name):
                    continue

                li = LineItem(name, amount, rate, qty if qty > 0 else 1.0)
                all_items.append(li)
                page_items_out.append(li.to_dict())
            except Exception:
                # skip malformed rows
                continue

        pagewise_output.append(
            {
                "page_no": str(idx),
                "page_type": page_type,
                "bill_items": page_items_out,
            }
        )

    # 3) Deduplicate across pages
    deduped = deduplicate_items(all_items)
    total_item_count = len(deduped)

    # NOTE: Evaluator will compute totals from bill_items;
    # we don't need to return totals explicitly.

    return JSONResponse(
        content={
            "is_success": True,
            "token_usage": token_usage,
            "data": {
                "pagewise_line_items": pagewise_output,
                "total_item_count": total_item_count,
            },
        }
    )


@app.get("/")
def root():
    return {"message": "Bajaj Health Datathon API ready"}
