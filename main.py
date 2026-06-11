import asyncio
import json
import os
import re
import uuid
from typing import AsyncGenerator

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from playwright.async_api import async_playwright
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

# ── In-memory job store ──────────────────────────────────────────────────────
jobs: dict[str, dict] = {}


class JobRequest(BaseModel):
    asins: list[str]


# ── Helpers ──────────────────────────────────────────────────────────────────

def asin_from_input(raw: str) -> str:
    """Extract ASIN from a URL or return as-is if already an ASIN."""
    match = re.search(r"/dp/([A-Z0-9]{10})", raw)
    if match:
        return match.group(1)
    raw = raw.strip().upper()
    if re.match(r"^[A-Z0-9]{10}$", raw):
        return raw
    raise ValueError(f"Cannot parse ASIN from: {raw}")


async def scrape_amazon(asin: str, page) -> dict:
    """Scrape product data and images from Amazon listing."""
    url = f"https://www.amazon.com/dp/{asin}"
    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
    await page.wait_for_timeout(2000)

    # Title
    title = ""
    try:
        title = await page.locator("#productTitle").inner_text(timeout=5000)
        title = title.strip()
    except Exception:
        pass

    # Price
    price = ""
    for selector in ["#priceblock_ourprice", ".a-price .a-offscreen", "#price_inside_buybox"]:
        try:
            price = await page.locator(selector).first.inner_text(timeout=3000)
            price = price.strip()
            if price:
                break
        except Exception:
            pass

    # Bullet points
    bullets = []
    try:
        items = await page.locator("#feature-bullets li span.a-list-item").all()
        for item in items:
            text = await item.inner_text()
            text = text.strip()
            if text:
                bullets.append(text)
    except Exception:
        pass

    # Description
    description = ""
    try:
        description = await page.locator("#productDescription p").first.inner_text(timeout=5000)
        description = description.strip()
    except Exception:
        pass

    # Images — extract from imageGalleryData or colorImages JS var
    images = []
    try:
        content = await page.content()
        # Try to find high-res image URLs in page JS
        matches = re.findall(r'"hiRes":"(https://[^"]+)"', content)
        if not matches:
            matches = re.findall(r'"large":"(https://[^"]+)"', content)
        seen = set()
        for m in matches:
            if m not in seen and "images/I/" in m:
                seen.add(m)
                images.append(m)
    except Exception:
        pass

    # Fallback: grab main image src
    if not images:
        try:
            src = await page.locator("#landingImage").get_attribute("src")
            if src:
                images.append(src)
        except Exception:
            pass

    return {
        "asin": asin,
        "title": title,
        "price": price,
        "bullets": bullets[:6],
        "description": description,
        "images": images[:8],
    }


async def scrape_tiktok_keywords(category: str) -> list[str]:
    """Scrape trending keywords from TikTok Creative Center."""
    keywords = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(
                f"https://ads.tiktok.com/business/creativecenter/keyword/pc/en",
                wait_until="domcontentloaded",
                timeout=20000,
            )
            await page.wait_for_timeout(3000)
            # Try to get keyword items
            items = await page.locator(".keyword-item, .trending-keyword, [class*='keyword']").all()
            for item in items[:20]:
                text = await item.inner_text()
                text = text.strip()
                if text and len(text) < 50:
                    keywords.append(text)
            await browser.close()
    except Exception:
        pass

    # Fallback: category-based keywords if scraping fails
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
        for key in fallback:
            if key in cat_lower:
                keywords = fallback[key]
                break
        if not keywords:
            keywords = fallback["default"]

    return keywords


def detect_category(title: str) -> str:
    """Detect product category from title."""
    title_lower = title.lower()
    categories = {
        "electronics": ["phone", "laptop", "tablet", "cable", "charger", "bluetooth", "wireless", "usb", "led", "camera", "speaker", "headphone", "earphone"],
        "home": ["pillow", "blanket", "kitchen", "storage", "organizer", "shelf", "lamp", "curtain", "mat", "towel", "furniture"],
        "beauty": ["serum", "moisturizer", "cream", "lipstick", "mascara", "foundation", "skincare", "hair", "nail", "perfume", "lotion"],
        "clothing": ["shirt", "dress", "pants", "shoes", "jacket", "hoodie", "socks", "hat", "bag", "wallet"],
        "fitness": ["dumbbell", "yoga", "resistance", "protein", "supplement", "gym", "workout", "exercise", "band"],
        "toys": ["toy", "game", "puzzle", "kids", "children", "play", "lego", "doll"],
    }
    for cat, keywords in categories.items():
        if any(kw in title_lower for kw in keywords):
            return cat
    return "general"


def enrich_with_groq(product: dict, tiktok_keywords: list[str]) -> dict:
    """Use Groq to rewrite listing for TikTok Shop."""
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
- Description: 3-4 sentences, conversational, include social proof language ("everyone's talking about"), end with soft CTA
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
    # Extract JSON
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    raise ValueError("Groq did not return valid JSON")


def generate_image_prompts(product: dict, images: list[str]) -> list[dict]:
    """Generate a ChatGPT prompt for each image."""
    category = detect_category(product["title"])
    title = product["title"]

    style_map = {
        "electronics": {
            "bg": "clean white background with subtle tech-grid shadow",
            "frame": "no frame, minimal clean edges",
            "angle": "slight 3/4 angle showing depth and ports",
        },
        "home": {
            "bg": "soft neutral lifestyle background, warm tones",
            "frame": "thin white border with soft drop shadow",
            "angle": "flat lay or slight elevated angle",
        },
        "beauty": {
            "bg": "pastel gradient background, pink or lavender",
            "frame": "elegant thin gold or white border",
            "angle": "front-facing with slight tilt for elegance",
        },
        "clothing": {
            "bg": "clean white or light grey background",
            "frame": "no frame, product fills frame",
            "angle": "front flat lay, evenly lit",
        },
        "fitness": {
            "bg": "energetic dark background or gym floor texture",
            "frame": "no frame, bold product presentation",
            "angle": "dynamic angle showing product in use context",
        },
        "default": {
            "bg": "clean white background, professional product shot",
            "frame": "thin white border with soft shadow",
            "angle": "front-facing, centred",
        },
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
4. REMOVE any barcodes, ISBNs, UPC codes, Amazon logos, or watermarks visible on the product or packaging
5. LIGHTING: Bright, even, commercial product lighting — no harsh shadows
6. OUTPUT SIZE: 800x800px square
7. Keep the product sharp and centred
8. Do not add any text or overlays

Return only the edited image file."""

        prompts.append({
            "image_index": i + 1,
            "image_url": img_url,
            "prompt": prompt_text,
        })

    return prompts


# ── SSE Pipeline ─────────────────────────────────────────────────────────────

async def run_pipeline(job_id: str, asins: list[str]) -> AsyncGenerator:
    """Run the full pipeline and stream progress events."""

    async def emit(event: str, data: dict):
        jobs[job_id]["events"].append({"event": event, "data": data})
        yield {"event": event, "data": json.dumps(data)}

    jobs[job_id] = {"status": "running", "events": [], "products": []}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        for idx, raw in enumerate(asins):
            try:
                asin = asin_from_input(raw)
            except ValueError as e:
                async for evt in emit("error", {"asin": raw, "message": str(e)}):
                    yield evt
                continue

            # Step 1: Scrape Amazon
            async for evt in emit("progress", {"asin": asin, "step": "scraping_amazon", "message": f"Scraping Amazon listing for {asin}..."}):
                yield evt

            try:
                product = await scrape_amazon(asin, page)
            except Exception as e:
                async for evt in emit("error", {"asin": asin, "message": f"Amazon scrape failed: {str(e)}"}):
                    yield evt
                continue

            # Step 2: TikTok keywords
            async for evt in emit("progress", {"asin": asin, "step": "fetching_keywords", "message": "Fetching TikTok trending keywords..."}):
                yield evt

            category = detect_category(product["title"])
            tiktok_keywords = await scrape_tiktok_keywords(category)

            # Step 3: Groq enrichment
            async for evt in emit("progress", {"asin": asin, "step": "enriching", "message": "Enriching listing with TikTok keywords via AI..."}):
                yield evt

            try:
                enriched = enrich_with_groq(product, tiktok_keywords)
            except Exception as e:
                async for evt in emit("error", {"asin": asin, "message": f"Enrichment failed: {str(e)}"}):
                    yield evt
                enriched = {
                    "title": product["title"],
                    "bullets": product["bullets"],
                    "description": product["description"],
                }

            # Step 4: Generate image prompts
            async for evt in emit("progress", {"asin": asin, "step": "generating_prompts", "message": "Generating ChatGPT image prompts..."}):
                yield evt

            image_prompts = generate_image_prompts(product, product["images"])

            # Final result
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

        await browser.close()

    jobs[job_id]["status"] = "done"
    async for evt in emit("done", {"message": "All products processed.", "total": len(asins)}):
        yield evt


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/api/jobs")
async def create_job(req: JobRequest):
    if not req.asins:
        raise HTTPException(status_code=400, detail="No ASINs provided")
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "events": [], "products": []}
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def generator():
        asins = jobs[job_id].get("asins", [])
        async for event in run_pipeline(job_id, asins):
            yield event

    return EventSourceResponse(generator())


@app.post("/api/jobs/{job_id}/start")
async def start_job(job_id: str, req: JobRequest):
    """Start processing — stores ASINs and streams via SSE."""
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
