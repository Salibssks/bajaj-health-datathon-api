# Bajaj Health Datathon API

            -- Approach--

- Convert input URL to bytes; treat as PDF (via PyMuPDF) or image (via Pillow).
- For each page:
      Render to high‑resolution PNG.
      Send base64 image + strict JSON schema prompt to OpenAI vision model.
-LLM returns page_type and bill_items.

           -- Backend: -- 

- Drops rows whose description matches summary/total keywords.
- Normalizes numbers, sets quantity 1 if missing but amount present.
- Deduplicates items across pages using fuzzy match on name + amount.

          -- Returns: --

- pagewise_line_items: list per page, with page_no, page_type, and bill_items.
- total_item_count: count of unique items after deduplication.


          -- Deployed API -- 

Render base URL - 
https://bajaj-health-datathon-api.onrender.com

Endpoint:
POST /extract-bill-data


GitHub repo: link ready.

Documentation: README explains approach + run instructions.
