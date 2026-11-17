from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import date
import os, json, re, io, requests
from pypdf import PdfReader

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("❌ مفيش GROQ_API_KEY في Environment Variables")

app = FastAPI(title="More English Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_API_KEYS = {"more-english-key-2024"}

def verify_key(api_key: str = Header(...)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="API Key غلط")
    return api_key

def extract_json(text):
    if isinstance(text, (list, dict)):
        return text
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"مقدرناش نستخرج JSON من الرد: {text}")

def ask_groq(prompt: str):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that only outputs valid JSON. No extra text, no markdown, just pure JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 2000
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"خطأ من Groq: {response.text}")
    result = response.json()
    return result["choices"][0]["message"]["content"]

def extract_text_from_pdf(content: bytes):
    try:
        reader = PdfReader(io.BytesIO(content))
        pdf_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"مقدرناش نقرا الـ PDF: {e}")
    if not pdf_text.strip():
        raise HTTPException(status_code=400, detail="ملف الـ PDF مفيهوش نص مقروء (ممكن يكون صور).")
    return pdf_text

def fetch_pdf_from_url(url: str):
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="مقدرناش نحمل الـ PDF من الـ URL")
        return extract_text_from_pdf(response.content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"خطأ في تحميل الـ PDF: {e}")


# ============================================================
# 1. أداة الفلاش كاردز — رفع ملف
# ============================================================
@app.post("/flashcards/generate")
async def generate_flashcards(
    file: UploadFile = File(...),
    api_key: str = Header(...)
):
    verify_key(api_key)
    content = await file.read()
    pdf_text = extract_text_from_pdf(content)

    prompt = f"""
    استخرج كل الكلمات الإنجليزية من النص ده مع معناها بالعربي.
    ارجع النتيجة بالشكل ده بس بدون أي كلام تاني (JSON Array فقط):
    [
        {{"english": "word", "arabic": "معنى"}},
        {{"english": "word", "arabic": "معنى"}}
    ]

    النص:
    {pdf_text[:3000]}
    """

    try:
        ai_response = ask_groq(prompt)
        cards = extract_json(ai_response)
        return {"flashcards": cards, "total": len(cards)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"رد غير صالح من Groq: {e}")


# ============================================================
# 1b. أداة الفلاش كاردز — عن طريق URL
# ============================================================
class UrlRequest(BaseModel):
    url: str

@app.post("/flashcards/generate-from-url")
async def generate_flashcards_from_url(
    data: UrlRequest,
    api_key: str = Header(...)
):
    verify_key(api_key)
    pdf_text = fetch_pdf_from_url(data.url)

    prompt = f"""
    استخرج كل الكلمات الإنجليزية من النص ده مع معناها بالعربي.
    ارجع النتيجة بالشكل ده بس بدون أي كلام تاني (JSON Array فقط):
    [
        {{"english": "word", "arabic": "معنى"}},
        {{"english": "word", "arabic": "معنى"}}
    ]

    النص:
    {pdf_text[:3000]}
    """

    try:
        ai_response = ask_groq(prompt)
        cards = extract_json(ai_response)
        return {"flashcards": cards, "total": len(cards)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"رد غير صالح من Groq: {e}")


# ============================================================
# 2. أداة الكويز — رفع ملف
# ============================================================
@app.post("/quiz/generate")
async def generate_quiz(
    file: UploadFile = File(...),
    api_key: str = Header(...)
):
    verify_key(api_key)
    content = await file.read()
    pdf_text = extract_text_from_pdf(content)

    prompt = f"""
    استخرج الأسئلة والإجابات من هذا الملف.
    ارجع النتيجة بالشكل ده بس بدون أي كلام تاني (JSON Array فقط):
    [
        {{
            "question": "السؤال هنا",
            "options": ["اختيار 1", "اختيار 2", "اختيار 3", "اختيار 4"],
            "correct_answer": "الإجابة الصح"
        }}
    ]

    النص:
    {pdf_text[:3000]}
    """

    try:
        ai_response = ask_groq(prompt)
        questions = extract_json(ai_response)
        return {"questions": questions, "total": len(questions)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"رد غير صالح من Groq: {e}")


# ============================================================
# 2b. أداة الكويز — عن طريق URL
# ============================================================
@app.post("/quiz/generate-from-url")
async def generate_quiz_from_url(
    data: UrlRequest,
    api_key: str = Header(...)
):
    verify_key(api_key)
    pdf_text = fetch_pdf_from_url(data.url)

    prompt = f"""
    استخرج الأسئلة والإجابات من هذا الملف.
    ارجع النتيجة بالشكل ده بس بدون أي كلام تاني (JSON Array فقط):
    [
        {{
            "question": "السؤال هنا",
            "options": ["اختيار 1", "اختيار 2", "اختيار 3", "اختيار 4"],
            "correct_answer": "الإجابة الصح"
        }}
    ]

    النص:
    {pdf_text[:3000]}
    """

    try:
        ai_response = ask_groq(prompt)
        questions = extract_json(ai_response)
        return {"questions": questions, "total": len(questions)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"رد غير صالح من Groq: {e}")


# ============================================================
# 3. أداة خطة المذاكرة
# ============================================================
class Lecture(BaseModel):
    id: str
    title: str
    date: str
    completed: bool
    quiz_score: Optional[float] = None

class StudentData(BaseModel):
    student_name: str
    lectures: List[Lecture]
    available_hours_per_day: float
    exam_date: str

@app.post("/study-plan/generate")
async def generate_study_plan(
    data: StudentData,
    api_key: str = Header(...)
):
    verify_key(api_key)

    today = date.today().isoformat()

    lectures_info = "\n".join([
        f"- {l.title} | تاريخ: {l.date} | خلص: {'نعم' if l.completed else 'لا'} | درجة الكويز: {l.quiz_score or 'لم يحل'}"
        for l in data.lectures
    ])

    prompt = f"""
    أنت مساعد تعليمي. اعمل خطة مذاكرة للطالب {data.student_name}.

    المعلومات:
    - تاريخ اليوم: {today}
    - تاريخ الامتحان: {data.exam_date}
    - ساعات متاحة في اليوم: {data.available_hours_per_day}

    المحاضرات:
    {lectures_info}

    القاعدة:
    - الجديد = محاضرات الأسبوع الحالي = 70% من وقت المذاكرة
    - القديم = محاضرات قبل الأسبوع الحالي = 30% من وقت المذاكرة
    - لو درجة الكويز أقل من 60% زود وقت المراجعة على المحاضرة دي

    ارجع الخطة بالشكل ده بس بدون أي كلام تاني (JSON Object فقط):
    {{
        "daily_plan": [
            {{
                "date": "YYYY-MM-DD",
                "sessions": [
                    {{
                        "lecture": "اسم المحاضرة",
                        "type": "جديد أو مراجعة",
                        "duration_minutes": 45,
                        "pomodoros": 1
                    }}
                ]
            }}
        ],
        "summary": "ملخص الخطة"
    }}
    """

    try:
        ai_response = ask_groq(prompt)
        plan = extract_json(ai_response)
        return plan
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"رد غير صالح من Groq: {e}")


# ============================================================
# 4. تجربة السيرفر
# ============================================================
@app.get("/")
def root():
    return {"message": "More English Backend شغال بـ Groq ✅ - كل الأدوات جاهزة"}
