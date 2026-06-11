import asyncio
import json
import os
import re
import uuid
from typing import AsyncGenerator

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

load_dotenv()

app = FastAPI(title="TikTok Dropship Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

jobs: dict[str, dict] = {}

AMAZON_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
}


class JobRequest(BaseModel):
    asins: list[str]


def asin_from_input(raw: str) -> str:
    match = re.search(r"/dp/([A-Z0-9]{10})", raw)
    if match:
        return match.group(1)
    raw = raw.strip().upper()
    if re.match(r"^[A-Z0-9]{10}$", raw):
        return raw
    raise ValueError(f"Cannot parse ASIN from: {raw}")


async def scrape_amazon(asin: str) -> dict:
    url = f"https://www.amazon.com/dp/{asin}"
    async with httpx.AsyncClient(headers=AMAZON_HEADERS, follow_redirects=True, timeout=20) as client:
        r = await client.get(url)
        r.raise_for_status()
        html = r.text

    soup = BeautifulSoup(html, "lxml")

    # Title
    title = ""
    el = soup.select_one("#productTitle")
    if el:
        title = el.get_text(strip=True)

    # Price
    price = ""
    for sel in ["#priceblock_ourprice", ".a-price .a-offscreen", "#price_inside_buybox", ".a-price-whole"]:
        el = soup.select_one(sel)
        if el:
            price = el.get_text(strip=True)
            break

    # Bullets
    bullets = []
    for li in soup.select("#feature-bullets li span.a-list-item"):
        t = li.get_text(strip=True)
        if t:
            bullets.append(t)

    # Description
    description = ""
    el = soup.select_one("#productDescription p")
    if el:
        description = el.get_text(strip=True)

    # Images from JS data
    images = []
    matches = re.findall(r'"hiRes"\s*:\s*"(https://[^"]+)"', html)
    if not matches:
        matches = re.findall(r'"large"\s*:\s*"(https://[^"]+)"', html)
    seen = set()
    for m in matches:
        if m not in seen and "images/I/" in m:
            seen.add(m)
            images.append(m)

    # Fallback main image
    if not images:
        el = soup.select_one("#landingImage")
        if el and el.get("src"):
            images.append(el["src"])
        elif el and el.get("data-src"):
            images.append(el["data-src"])

    return {
        "asin": asin,
        "title": title or f"Product {asin}",
        "price": price,
        "bullets": bullets[:6],
        "description": description,
        "images": images[:8],
    }


async def scrape_tiktok_keywords(category: str) -> list[str]:
    """Fetch trending keywords from TikTok Creative Center."""
    keywords = []
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            r = await client.get(
                "https://ads.tiktok.com/business/creativecenter/keyword/pc/en",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            soup = BeautifulSoup(r.text, "lxml")
            for el in soup.select("[class*='keyword']")[:20]:
                t = el.get_text(strip=True)
                if t and 2 < len(t) < 40:
                    keywords.append(t)
    except Exception:
        pass

    if not keywords:
        fallback = {
            "electronics": ["viral gadget", "tech deal", "must have", "life hack", "under $20", "aesthetic", "trending tech"],
            "home": ["home decor", "aesthetic room", "organization", "clean girl", "cozy home", "apartment essentials"],
            "beauty": ["skincare routine", "glow up", "beauty hack", "viral beauty", "must have", "self care"],
            "clothing": ["outfit inspo", "aesthetic fit", "viral fashion", "OOTD", "style hack", "trendy"],
            "fitness": ["gym essentials", "workout gear", "fitness hack", "health tips", "body goals"],
            "default": ["viral product", "must have", "TikTok made me buy it", "life changing", "under $30", "aesthetic"],
        }
        cat_lower = category.lower()
        keywords = next((v for k, v in fallback.items() if k in cat_lower), fallback["default"])

    return keywords


def detect_category(title: str) -> str:
    title_lower = title.lower()
    cats = {
        "electronics": ["phone", "laptop", "tablet", "cable", "charger", "bluetooth", "wireless", "usb", "led", "camera", "speaker", "headphone", "earphone"],
        "home": ["pillow", "blanket", "kitchen", "storage", "organizer", "shelf", "lamp", "curtain", "mat", "towel"],
        "beauty": ["serum", "moisturizer", "cream", "lipstick", "mascara", "foundation", "skincare", "hair", "nail", "perfume"],
        "clothing": ["shirt", "dress", "pants", "shoes", "jacket", "hoodie", "socks", "hat", "bag", "wallet"],
        "fitness": ["dumbbell", "yoga", "resistance", "protein", "supplement", "gym", "workout", "exercise"],
        "toys": ["toy", "game", "puzzle", "kids", "children", "play", "lego", "doll"],
    }
    for cat, keywords in cats.items():
        if any(kw in title_lower for kw in keywords):
            return cat
    return "general"


def enrich_with_groq(product: dict, tiktok_keywords: list[str]) -> dict:
    kw_str = ", ".join(tiktok_keywords[:10])
    bullets_str = "\n".join(f"- {b}" for b in product["bullets"])

    prompt = f"""You are a TikTok Shop listing expert. Rewrite this Amazon product listing for TikTok Shop.

PRODUCT DATA:
Title: {product['title']}
Price: {product['price']}
Bullets:
{bullets_str}
Description: {product['description'][:500]}

TRENDING TIKTOK KEYWORDS TO NATURALLY INCLUDE: {kw_str}

RULES:
- Title: max 60 chars, include 2-3 trending keywords naturally, benefit-focused
- Bullets: 5 bullets, short punchy sentences, benefits over features, TikTok Gen-Z tone
- Description: 3-4 sentences, conversational, include social proof language, end with soft CTA
- Do NOT sound like Amazon. Sound like a TikTok creator recommending a product.

Respond ONLY in this exact JSON format:
{{
  "title": "...",
  "bullets": ["...", "...", "...", "...", "..."],
  "description": "..."
}}"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=800,
    )

    text = response.choices[0].message.content.strip()
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    raise ValueError("Groq did not return valid JSON")


def generate_image_prompts(product: dict, images: list[str]) -> list[dict]:
    category = detect_category(product["title"])
    title = product["title"]

    style_map = {
        "electronics": {"bg": "clean white background with subtle tech-grid shadow", "frame": "no frame, minimal clean edges", "angle": "slight 3/4 angle showing depth"},
        "home": {"bg": "soft neutral lifestyle background, warm tones", "frame": "thin white border with soft drop shadow", "angle": "flat lay or slight elevated angle"},
        "beauty": {"bg": "pastel gradient background, pink or lavender", "frame": "elegant thin gold or white border", "angle": "front-facing with slight tilt"},
        "clothing": {"bg": "clean white or light grey background", "frame": "no frame, product fills frame", "angle": "front flat lay, evenly lit"},
        "fitness": {"bg": "energetic dark background or gym floor texture", "frame": "no frame, bold product presentation", "angle": "dynamic angle"},
        "default": {"bg": "clean white background, professional product shot", "frame": "thin white border with soft shadow", "angle": "front-facing, centred"},
    }

    style = style_map.get(category, style_map["default"])
    prompts = []

    for i, img_url in enumerate(images):
        prompt_text = f"""Edit this product image for TikTok Shop listing.

PRODUCT: {title}

INSTRUCTIONS:
1. BACKGROUND: Replace background with {style['bg']}
2. FRAME: {style['frame']}
3. ANGLE/PERSPECTIVE: {style['angle']}
4. REMOVE any barcodes, ISBNs, UPC codes, Amazon logos, or watermarks
5. LIGHTING: Bright, even, commercial product lighting — no harsh shadows
6. OUTPUT SIZE: 800x800px square
7. Keep the product sharp and centred
8. Do not add any text or overlays

Return only the edited image file."""

        prompts.append({"image_index": i + 1, "image_url": img_url, "prompt": prompt_text})

    return prompts


async def run_pipeline(job_id: str, asins: list[str]) -> AsyncGenerator:
    async def emit(event: str, data: dict):
        jobs[job_id]["events"].append({"event": event, "data": data})
        yield {"event": event, "data": json.dumps(data)}

    jobs[job_id] = {"status": "running", "events": [], "products": []}

    for raw in asins:
        try:
            asin = asin_from_input(raw)
        except ValueError as e:
            async for evt in emit("error", {"asin": raw, "message": str(e)}):
                yield evt
            continue

        async for evt in emit("progress", {"asin": asin, "step": "scraping_amazon", "message": f"Scraping Amazon listing for {asin}..."}):
            yield evt

        try:
            product = await scrape_amazon(asin)
        except Exception as e:
            async for evt in emit("error", {"asin": asin, "message": f"Amazon scrape failed: {str(e)}"}):
                yield evt
            continue

        async for evt in emit("progress", {"asin": asin, "step": "fetching_keywords", "message": "Fetching TikTok trending keywords..."}):
            yield evt

        category = detect_category(product["title"])
        tiktok_keywords = await scrape_tiktok_keywords(category)

        async for evt in emit("progress", {"asin": asin, "step": "enriching", "message": "Enriching listing with AI..."}):
            yield evt

        try:
            enriched = enrich_with_groq(product, tiktok_keywords)
        except Exception as e:
            async for evt in emit("error", {"asin": asin, "message": f"Enrichment failed: {str(e)}"}):
                yield evt
            enriched = {"title": product["title"], "bullets": product["bullets"], "description": product["description"]}

        async for evt in emit("progress", {"asin": asin, "step": "generating_prompts", "message": "Generating ChatGPT image prompts..."}):
            yield evt

        image_prompts = generate_image_prompts(product, product["images"])

        result = {
            "asin": asin,
            "original": product,
            "enriched": enriched,
            "tiktok_keywords": tiktok_keywords,
            "image_prompts": image_prompts,
            "category": category,
        }
        jobs[job_id]["products"].append(result)

        async for evt in emit("product_done", {"asin": asin, "product": result}):
            yield evt

    jobs[job_id]["status"] = "done"
    async for evt in emit("done", {"message": "All products processed.", "total": len(asins)}):
        yield evt


@app.post("/api/jobs")
async def create_job(req: JobRequest):
    if not req.asins:
        raise HTTPException(status_code=400, detail="No ASINs provided")
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "events": [], "products": []}
    return {"job_id": job_id}


@app.post("/api/jobs/{job_id}/start")
async def start_job(job_id: str, req: JobRequest):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    jobs[job_id]["asins"] = req.asins

    async def generator():
        async for event in run_pipeline(job_id, req.asins):
            yield event

    return EventSourceResponse(generator())


@app.get("/api/jobs/{job_id}/results")
async def get_results(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/api/jobs/all")
async def get_all_jobs():
    return jobs


@app.get("/health")
async def health():
    return {"status": "ok"}
