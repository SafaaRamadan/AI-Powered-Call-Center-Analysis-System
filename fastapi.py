from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import nemo.collections.asr as nemo_asr
import json
import os
import tempfile8
import librosa
import numpy as np
from datetime import datetime
import logging
from functools import lru_cache

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Call Center AI Agent API - EgypTalk ASR Version",
    version="2.0.0-egyptalk",
    description="Production-ready microservice using EgypTalk ASR v2 for Egyptian Arabic"
)

# CORS for .NET Backend
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONFIGURATION (ALL FREE - NO API KEYS NEEDED)
# ============================================================================

class Config:
    # Model Settings - EgypTalk ASR (Best for Egyptian Arabic!)
    ASR_MODEL = os.getenv("ASR_MODEL", "NAMAA-Space/EgypTalk-ASR-v2")
    USE_GPU = torch.cuda.is_available()
    
    # Local LLM Settings (FREE alternatives)
    USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"
    LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "facebook/opt-1.3b")
    
    # Audio Analysis Thresholds
    RMS_THRESHOLD_HIGH = float(os.getenv("RMS_THRESHOLD_HIGH", "0.15"))
    RMS_THRESHOLD_CRITICAL = float(os.getenv("RMS_THRESHOLD_CRITICAL", "0.25"))
    
    # Processing
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/call_analysis")

config = Config()

# Create temp directory
os.makedirs(config.TEMP_DIR, exist_ok=True)

logger.info(f"🚀 FREE VERSION - No API costs!")
logger.info(f"Config: GPU={config.USE_GPU}, Local LLM={config.USE_LOCAL_LLM}, ASR={config.ASR_MODEL}")

# ============================================================================
# RESPONSE MODELS (Pydantic)
# ============================================================================

class ViolationDetail(BaseModel):
    word: str
    position: int
    type: str
    context: str
    timestamp: Optional[str] = None

class CriteriaAnalysis(BaseModel):
    criteriaName: str
    score: float
    violations: List[ViolationDetail]
    severity: str
    suggestions: str

class EmotionAnalysis(BaseModel):
    emotionLabel: str
    confidenceScore: float
    reasoning: str

class LLMInsights(BaseModel):
    overallAssessment: str
    strengths: List[str]
    weaknesses: List[str]
    criticalMoments: List[Dict]
    improvementSuggestions: List[str]
    problemResolutionScore: float

class AnalysisReport(BaseModel):
    transcript: str
    overallScore: float
    processingTime: float
    criteriaScores: List[CriteriaAnalysis]
    emotionResult: EmotionAnalysis
    llmInsights: LLMInsights
    analyzedAt: str

# ============================================================================
# AI MODELS INITIALIZATION (100% FREE - NO API COSTS)
# ============================================================================

@lru_cache(maxsize=1)
def load_stt_model():
    """Load Speech-to-Text model (FREE - EgypTalk ASR v2 for Egyptian Arabic)"""
    try:
        logger.info(f"Loading EgypTalk ASR model: {config.ASR_MODEL}...")
        
        # Load EgypTalk model using NeMo
        model = nemo_asr.models.ASRModel.from_pretrained(config.ASR_MODEL)
        
        # Move to GPU if available
        if config.USE_GPU:
            model = model.cuda()
        else:
            model = model.cpu()
        
        logger.info(f"✅ EgypTalk ASR loaded - Optimized for Egyptian Arabic dialect!")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load EgypTalk ASR: {e}")
        raise

@lru_cache(maxsize=1)
def load_emotion_model():
    """Load emotion classifier (FREE)"""
    try:
        logger.info("Loading emotion model...")
        model = pipeline(
            "audio-classification",
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            device=0 if config.USE_GPU else -1
        )
        logger.info("✅ Emotion model loaded")
        return model
    except Exception as e:
        logger.warning(f"⚠️ Emotion model not loaded: {e}")
        return None

@lru_cache(maxsize=1)
def load_text_sentiment():
    """Load Arabic sentiment model (FREE)"""
    try:
        logger.info("Loading Arabic sentiment model...")
        model = pipeline(
            "sentiment-analysis",
            model="CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment",
            device=0 if config.USE_GPU else -1
        )
        logger.info("✅ Sentiment model loaded")
        return model
    except Exception as e:
        logger.warning(f"⚠️ Sentiment model not loaded: {e}")
        return None

@lru_cache(maxsize=1)
def load_local_llm():
    """Load LOCAL LLM for text generation (FREE - no API costs)"""
    if not config.USE_LOCAL_LLM:
        logger.info("Local LLM disabled")
        return None, None
    
    try:
        logger.info(f"Loading local LLM: {config.LOCAL_LLM_MODEL}...")
        device = "cuda" if config.USE_GPU else "cpu"
        
        tokenizer = AutoTokenizer.from_pretrained(config.LOCAL_LLM_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            config.LOCAL_LLM_MODEL,
            torch_dtype=torch.float16 if config.USE_GPU else torch.float32,
            device_map="auto" if config.USE_GPU else None
        )
        
        if not config.USE_GPU:
            model = model.to(device)
        
        logger.info(f"✅ Local LLM loaded on {device}")
        return tokenizer, model
    except Exception as e:
        logger.warning(f"⚠️ Local LLM not loaded: {e}")
        return None, None

# Initialize models (lazy loading)
logger.info("🔄 Models will be loaded on first request (FREE - No API costs)...")
stt_model = None
emotion_classifier = None
text_sentiment = None
local_llm_tokenizer = None
local_llm_model = None

def ensure_models_loaded():
    """Ensure all FREE models are loaded"""
    global stt_model, emotion_classifier, text_sentiment, local_llm_tokenizer, local_llm_model
    
    if stt_model is None:
        stt_model = load_stt_model()
    if emotion_classifier is None:
        emotion_classifier = load_emotion_model()
    if text_sentiment is None:
        text_sentiment = load_text_sentiment()
    if local_llm_tokenizer is None and config.USE_LOCAL_LLM:
        local_llm_tokenizer, local_llm_model = load_local_llm()

# ============================================================================
# NLP ANALYZER CLASS
# ============================================================================

class AgentAnalyzer:
    def __init__(self):
        self.slang_words = self._load_slang_words()
        self.uncertainty_phrases = self._load_uncertainty_phrases()
        self.disrespectful = self._load_disrespectful_words()
    
    def _load_slang_words(self):
        """Load slang words (can be loaded from config/DB)"""
        return [
            "يسطا", "بص", "يعم", "ماشي", "حلو", "كدا", "خلاص", "يا عم",
            "هو انت", "ايه يعني", "خلاص بقى", "طب و", "يلا", "هاتلي",
            "شوف", "قولي", "ايوة", "لا يا عم", "معلش", "سيبك",
            "براحتك", "عادي", "مش مهم", "متقلقش"
        ]
    
    def _load_uncertainty_phrases(self):
        """Load uncertainty phrases"""
        return [
            "معرفش", "مش متأكد", "مش عارف", "مفيش فكرة", 
            "مش واضح", "ممكن", "يمكن", "I don't know",
            "مش قادر أحدد", "صعب أقول", "محتاج أتأكد",
            "لازم أشوف", "هشوف وأرجعلك", "مش من اختصاصي"
        ]
    
    def _load_disrespectful_words(self):
        """Load disrespectful words"""
        return [
            "اسكت", "مش فاهم حاجة", "غبي", "هبل", 
            "مش شغلك", "بلاش دماغ", "سيب دماغي",
            "انت مين", "بتعمل ايه", "مش عاجبك", "روح",
            "متزهقش", "خليك في حالك"
        ]
    
    def analyze_formality(self, text: str) -> Dict:
        """Detect informal/slang language"""
        violations = []
        text_lower = text.lower()
        
        for word in self.slang_words:
            if word in text_lower:
                position = text_lower.index(word)
                violations.append(ViolationDetail(
                    word=word,
                    position=position,
                    type="informal_language",
                    context=text[max(0, position-30):min(len(text), position+30)]
                ))
        
        score = max(0, 100 - (len(violations) * 12))
        severity = "high" if score < 60 else "medium" if score < 80 else "low"
        
        suggestions = self._generate_formality_suggestions(violations)
        
        return CriteriaAnalysis(
            criteriaName="Language Formality",
            score=score,
            violations=violations,
            severity=severity,
            suggestions=suggestions
        )
    
    def analyze_respect(self, text: str) -> Dict:
        """Detect disrespectful language"""
        violations = []
        text_lower = text.lower()
        
        for word in self.disrespectful:
            if word in text_lower:
                position = text_lower.index(word)
                violations.append(ViolationDetail(
                    word=word,
                    position=position,
                    type="disrespectful_language",
                    context=text[max(0, position-30):min(len(text), position+30)]
                ))
        
        score = max(0, 100 - (len(violations) * 25))
        severity = "critical" if score < 50 else "high" if score < 75 else "low"
        
        suggestions = "تجنب استخدام ألفاظ غير محترمة. استخدم لغة مهنية في جميع الأوقات."
        
        return CriteriaAnalysis(
            criteriaName="Respectful Communication",
            score=score,
            violations=violations,
            severity=severity,
            suggestions=suggestions
        )
    
    def analyze_uncertainty(self, text: str) -> Dict:
        """Detect uncertainty expressions"""
        violations = []
        text_lower = text.lower()
        
        for phrase in self.uncertainty_phrases:
            if phrase in text_lower:
                position = text_lower.index(phrase)
                violations.append(ViolationDetail(
                    word=phrase,
                    position=position,
                    type="uncertainty_expression",
                    context=text[max(0, position-30):min(len(text), position+30)]
                ))
        
        score = max(0, 100 - (len(violations) * 15))
        severity = "high" if score < 65 else "medium" if score < 85 else "low"
        
        suggestions = "تجنب التعبير عن عدم المعرفة. استخدم: 'سأتحقق وأعود إليك' أو 'دعني أبحث عن المعلومة'"
        
        return CriteriaAnalysis(
            criteriaName="Certainty & Confidence",
            score=score,
            violations=violations,
            severity=severity,
            suggestions=suggestions
        )
    
    def analyze_voice_level(self, audio_path: str) -> Dict:
        """Detect raised voice from audio"""
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            rms = librosa.feature.rms(y=y)[0]
            avg_rms = np.mean(rms)
            max_rms = np.max(rms)
            
            threshold_high = config.RMS_THRESHOLD_HIGH
            threshold_critical = config.RMS_THRESHOLD_CRITICAL
            
            violations = []
            high_moments = np.where(rms > threshold_critical)[0]
            
            if len(high_moments) > 0:
                for moment in high_moments[:3]:
                    timestamp = moment * 512 / sr
                    violations.append(ViolationDetail(
                        word="Raised Voice",
                        position=int(timestamp),
                        type="loud_voice",
                        context=f"Loud moment at ~{timestamp:.1f}s (RMS: {rms[moment]:.3f})",
                        timestamp=f"{int(timestamp//60)}:{int(timestamp%60):02d}"
                    ))
            
            if max_rms > threshold_critical:
                score = max(0, 100 - ((max_rms - threshold_critical) * 300))
            elif avg_rms > threshold_high:
                score = max(60, 100 - ((avg_rms - threshold_high) * 200))
            else:
                score = 100
            
            severity = "critical" if max_rms > threshold_critical else "medium" if avg_rms > threshold_high else "low"
            
            logger.info(f"Voice analysis: avg_rms={avg_rms:.3f}, max_rms={max_rms:.3f}, score={score}")
            
            return CriteriaAnalysis(
                criteriaName="Voice Level Control",
                score=score,
                violations=violations,
                severity=severity,
                suggestions="حافظ على نبرة صوت هادئة ومحترمة طوال المكالمة"
            )
        except Exception as e:
            logger.error(f"Voice analysis failed: {e}")
            return CriteriaAnalysis(
                criteriaName="Voice Level Control",
                score=100,
                violations=[],
                severity="low",
                suggestions="Unable to analyze voice level"
            )
    
    def _generate_formality_suggestions(self, violations: List) -> str:
        if not violations:
            return "أداء ممتاز في استخدام اللغة الرسمية"
        
        examples = {
            "يسطا": "استخدم: 'سيدي' أو 'حضرتك'",
            "بص": "استخدم: 'من فضلك راجع' أو 'دعني أوضح'",
            "يعم": "استخدم: 'سيدي' أو الاسم المباشر",
        }
        
        suggestions = []
        for v in violations[:3]:
            word = v.word
            if word in examples:
                suggestions.append(examples[word])
        
        return " | ".join(suggestions) if suggestions else "استخدم لغة رسمية واضحة"

class EmotionAnalyzer:
    def __init__(self):
        self.audio_classifier = emotion_classifier
        self.text_classifier = text_sentiment
    
    def analyze(self, audio_path: str, transcript: str) -> EmotionAnalysis:
        """Analyze customer emotion from audio and text"""
        
        audio_emotion = "neutral"
        audio_confidence = 0.5
        
        if self.audio_classifier:
            try:
                result = self.audio_classifier(audio_path)
                if result:
                    audio_emotion = result[0]['label'].lower()
                    audio_confidence = result[0]['score']
            except:
                pass
        
        text_emotion = "neutral"
        text_confidence = 0.5
        
        if self.text_classifier:
            try:
                result = self.text_classifier(transcript[:512])
                if result:
                    sentiment = result[0]['label'].lower()
                    text_confidence = result[0]['score']
                    if 'pos' in sentiment:
                        text_emotion = "satisfied"
                    elif 'neg' in sentiment:
                        text_emotion = "frustrated"
            except:
                pass
        
        final_emotion, final_confidence, reasoning = self._fuse_emotions(
            audio_emotion, audio_confidence,
            text_emotion, text_confidence,
            transcript
        )
        
        return EmotionAnalysis(
            emotionLabel=final_emotion,
            confidenceScore=final_confidence,
            reasoning=reasoning
        )
    
    def _fuse_emotions(self, audio_emo, audio_conf, text_emo, text_conf, transcript):
        emotions = [audio_emo, text_emo]
        final = max(set(emotions), key=emotions.count)
        conf = (audio_conf + text_conf) / 2
        return final, conf, f"Based on audio ({audio_emo}) and text ({text_emo}) analysis"

class LLMEnhancer:
    def __init__(self):
        self.tokenizer = local_llm_tokenizer
        self.model = local_llm_model
    
    def enhance_analysis(self, transcript: str, nlp_results: List[CriteriaAnalysis]) -> LLMInsights:
        """Deep analysis with LOCAL LLM (100% FREE)"""
        
        if not config.USE_LOCAL_LLM or not self.model:
            logger.warning("Local LLM disabled, using rule-based insights")
            return self._rule_based_insights(nlp_results, transcript)
        
        nlp_summary = "\n".join([
            f"- {cr.criteriaName}: {cr.score}/100 ({len(cr.violations)} issues)"
            for cr in nlp_results
        ])
        
        transcript_preview = transcript[:800] if len(transcript) > 800 else transcript
        
        prompt = f"""Analyze this customer service call:

Call transcript (Arabic): {transcript_preview}

Analysis scores:
{nlp_summary}

Provide brief assessment in JSON:
{{"assessment": "brief evaluation", "strengths": ["point1"], "weaknesses": ["point1"], "suggestions": ["tip1"]}}"""

        try:
            logger.info("Running local LLM analysis...")
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            if config.USE_GPU:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            try:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    result = json.loads(response[json_start:json_end])
                else:
                    raise ValueError("No JSON found")
                
                logger.info(" Local LLM analysis complete")
                
                return LLMInsights(
                    overallAssessment=result.get('assessment', 'Analysis completed'),
                    strengths=result.get('strengths', ['Needs review']),
                    weaknesses=result.get('weaknesses', ['Needs review']),
                    criticalMoments=[],
                    improvementSuggestions=result.get('suggestions', ['Review with supervisor']),
                    problemResolutionScore=sum(cr.score for cr in nlp_results) / len(nlp_results)
                )
            except:
                logger.warning("Local LLM output parsing failed, using rules")
                return self._rule_based_insights(nlp_results, transcript)
        
        except Exception as e:
            logger.error(f"Local LLM failed: {e}, using rule-based insights")
            return self._rule_based_insights(nlp_results, transcript)
    
    def _rule_based_insights(self, nlp_results: List[CriteriaAnalysis], transcript: str) -> LLMInsights:
        """Rule-based insights when LLM unavailable"""
        avg_score = sum(cr.score for cr in nlp_results) / len(nlp_results)
        
        all_violations = []
        for cr in nlp_results:
            all_violations.extend([(cr.criteriaName, v) for v in cr.violations])
        
        weaknesses = []
        strengths = []
        suggestions = []
        
        for cr in nlp_results:
            if cr.score < 70:
                weaknesses.append(f"{cr.criteriaName}: {len(cr.violations)} مخالفات ({cr.severity})")
                suggestions.append(cr.suggestions)
            elif cr.score >= 90:
                strengths.append(f"{cr.criteriaName}: أداء ممتاز")
        
        if avg_score >= 85:
            assessment = f"أداء جيد جداً بمتوسط {avg_score:.1f}/100. الموظف ملتزم بمعايير الخدمة."
        elif avg_score >= 70:
            assessment = f"أداء جيد بمتوسط {avg_score:.1f}/100. يوجد مجال للتحسين في بعض النقاط."
        else:
            assessment = f"الأداء يحتاج تحسين بمتوسط {avg_score:.1f}/100. يجب مراجعة المعايير مع المشرف."
        
        problem_resolution = avg_score
        if "معرفش" in transcript or "مش متأكد" in transcript:
            problem_resolution -= 10
        if len(transcript) > 1000:
            problem_resolution = min(100, problem_resolution + 5)
        
        return LLMInsights(
            overallAssessment=assessment,
            strengths=strengths if strengths else ["يحتاج تطوير في جميع المعايير"],
            weaknesses=weaknesses if weaknesses else ["لا توجد مخالفات واضحة"],
            criticalMoments=[
                {"timestamp": "N/A", "issue": v.type, "context": v.context[:50]}
                for _, v in all_violations[:3]
            ],
            improvementSuggestions=suggestions if suggestions else ["الاستمرار على نفس المستوى"],
            problemResolutionScore=max(0, min(100, problem_resolution))
        )

analyzer = AgentAnalyzer()
emotion_analyzer = EmotionAnalyzer()
llm_enhancer = LLMEnhancer()

def analyze_call(audio_path: str) -> AnalysisReport:
    """Main pipeline: EgypTalk ASR → NLP → LLM → Emotion"""
    
    start_time = datetime.now()
    
    try:
        ensure_models_loaded()
        
        # 1. Speech to Text (EgypTalk ASR - Best for Egyptian Arabic!)
        logger.info(" Step 1/4: Transcribing audio with EgypTalk ASR v2...")
        transcription_result = stt_model.transcribe([audio_path])
        
        # Extract text from NeMo result
        if isinstance(transcription_result, list) and len(transcription_result) > 0:
            transcript = transcription_result[0]
        else:
            transcript = str(transcription_result)
        
        logger.info(f" Transcription complete ({len(transcript)} chars)")
        
        # 2. NLP Analysis
        logger.info(" Step 2/4: NLP Analysis...")
        formality = analyzer.analyze_formality(transcript)
        respect = analyzer.analyze_respect(transcript)
        uncertainty = analyzer.analyze_uncertainty(transcript)
        voice_level = analyzer.analyze_voice_level(audio_path)
        
        nlp_results = [formality, respect, uncertainty, voice_level]
        logger.info(f" NLP complete")
        
        # 3. LLM Enhancement
        logger.info(" Step 3/4: LLM Deep Analysis...")
        llm_insights = llm_enhancer.enhance_analysis(transcript, nlp_results)
        logger.info(f" LLM complete")
        
        # 4. Emotion Analysis
        logger.info(" Step 4/4: Emotion Detection...")
        emotion_result = emotion_analyzer.analyze(audio_path, transcript)
        logger.info(f" Emotion: {emotion_result.emotionLabel}")
        
        # 5. Calculate Overall Score
        criteria_scores_values = [cr.score for cr in nlp_results]
        overall_score = (
            sum(criteria_scores_values) * 0.6 +
            llm_insights.problemResolutionScore * 0.4
        ) / len(criteria_scores_values)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f" Analysis complete: Score={overall_score:.2f}, Time={processing_time:.2f}s")
        
        return AnalysisReport(
            transcript=transcript,
            overallScore=round(overall_score, 2),
            processingTime=round(processing_time, 2),
            criteriaScores=nlp_results,
            emotionResult=emotion_result,
            llmInsights=llm_insights,
            analyzedAt=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f" Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/")
def root():
    return {
        "service": "Call Center AI Agent API - EgypTalk ASR",
        "version": "2.0.0-egyptalk",
        "status": "running",
        "endpoints": {
            "analyze": "POST /api/analyze",
            "health": "GET /health"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0-egyptalk",
        "cost": " $0.00 - 100% FREE!",
        "asr": "EgypTalk ASR v2 - Egyptian Arabic Optimized",
        "models_loaded": {
            "stt": stt_model is not None,
            "emotion": emotion_classifier is not None,
            "sentiment": text_sentiment is not None,
            "local_llm": local_llm_model is not None
        },
        "gpu_available": config.USE_GPU
    }

@app.post("/api/analyze", response_model=AnalysisReport)
async def analyze_call_recording(
    audio: UploadFile = File(...),
    agent_id: Optional[int] = Form(None)
):
    """
    Main endpoint: Analyze call recording with EgypTalk ASR v2
    
    Parameters:
    - audio: Audio file (WAV, MP3, M4A, etc.) - Max 100MB
    - agent_id: Optional agent ID for tracking
    
    Returns:
    - Complete analysis report with scores, violations, and insights
    """
    
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    file_ext = os.path.splitext(audio.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Allowed: {allowed_extensions}"
        )
    
    content = await audio.read()
    file_size = len(content)
    
    if file_size > config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {config.MAX_FILE_SIZE / 1024 / 1024}MB"
        )
    
    temp_path = os.path.join(
        config.TEMP_DIR,
        f"{datetime.now().timestamp()}_{audio.filename}"
    )
    
    try:
        with open(temp_path, 'wb') as f:
            f.write(content)
        
        logger.info(f"Analyzing call: file={audio.filename}, size={file_size/1024:.1f}KB, agent_id={agent_id}")
        
        report = analyze_call(temp_path)
        
        logger.info(f"Analysis complete: Score={report.overallScore}")
        
        return report
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                logger.warning(f"Could not delete temp file: {temp_path}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)