from flask import Flask
from flask_restx import Api, Resource, fields
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
import nltk
from rouge_score import rouge_scorer
import sacrebleu
from rank_bm25 import BM25Okapi
import requests
import openai
import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging
import time
import traceback
import jieba
from bert_score import score as bert_score
import evaluate
from transformers import AutoTokenizer, AutoModel
import torch

# 載入環境變數
load_dotenv()

app = Flask(__name__)
api = Api(app, version='1.0', 
          title='向量檢索 API',
          description='支援多種檢索方法和評估指標的向量檢索 API',
          doc='/docs',
          default='retrieve',  # 設置默認命名空間
          ordered=True  # 保持參數順序
        )

# API 配置
NOCODB_API_URL = os.getenv('NOCODB_API_URL', 'https://mynocodb.zeabur.app/api/v2/tables/mno08ehjtvwt6w2/records')
NOCODB_API_KEY = os.getenv('NOCODB_API_KEY', 'aIPkBt4RVtZKtXxqcwDLeMaKlNUlGtlz2n3yNaOH')

# 模型配置
DEFAULT_EMBEDDING_MODEL = os.getenv('DEFAULT_EMBEDDING_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2')
DEFAULT_GPT_MODEL = os.getenv('DEFAULT_GPT_MODEL', 'gpt-4')
DEFAULT_RETRIEVER_METHOD = os.getenv('DEFAULT_RETRIEVER_METHOD', 'top_k')
DEFAULT_K_VALUE = int(os.getenv('DEFAULT_K_VALUE', '3'))

# OpenAI 配置
openai.api_key = os.getenv('OPENAI_API_KEY')

# Gemini 配置
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel(os.getenv('GEMINI_MODEL', 'gemini-2.0-flash'))

# 定義命名空間
ns = api.namespace('api', description='檢索操作')

# 下載必要的 NLTK 資料
nltk.download('punkt')

# API 請求模型定義
retrieve_request = api.model('RetrieveRequest', {
    'embedding_model': fields.String(
        description='使用的嵌入模型名稱，用於文本向量化。預設值：paraphrase-multilingual-MiniLM-L12-v2',
        required=False, 
        default=DEFAULT_EMBEDDING_MODEL
    ),
    'retriever_method': fields.String(
        description='檢索方法：\n'
                   '- top_k：直接返回最相似的 k 個文檔\n'
                   '- mmr：最大邊際相關性，在相關性和多樣性之間取得平衡',
        required=False,
        default=DEFAULT_RETRIEVER_METHOD,
        enum=['top_k', 'mmr']
    ),
    'k_value': fields.Integer(
        description='返回的文檔數量，建議範圍：1-10',
        required=False,
        default=DEFAULT_K_VALUE,
        min=1,
        max=10
    ),
    'query': fields.String(
        description='查詢文本，用戶想要查詢的問題或關鍵詞',
        required=True
    ),
    'model': fields.String(
        description='使用的生成模型：\n'
                   '- gpt-4：使用 OpenAI 的 GPT-4 模型\n'
                   '- gemini：使用 Google 的 Gemini 模型',
        required=False,
        default=DEFAULT_GPT_MODEL,
        enum=['gpt-4', 'gemini']
    ),
    'model_name': fields.String(
        description='指定的具體模型版本：\n'
                   '- GPT-4 系列：gpt-4, gpt-4-32k, gpt-4-turbo\n'
                   '- Gemini 系列：gemini-1.0-pro, gemini-1.0-ultra, gemini-2.0-flash, gemini-2.0-pro',
        required=False
    ),
    'temperature': fields.Float(
        description='生成模型的溫度參數 (0.0-1.0)，控制回答的創造性，越低越保守。建議範圍：0.2-0.5',
        required=False,
        default=0.2,
        min=0.0,
        max=1.0
    ),
    'prompt': fields.String(
        description='自定義的系統提示詞模板，用於自定義模型的行為指南和回答風格',
        required=False
    )
})

document_model = api.model('Document', {
    'document': fields.String(description='文檔內容'),
    'score': fields.Float(description='相似度分數')
})

metrics_model = api.model('Metrics', {
    'bert_score_p': fields.Float(
        description='BERTScore 精確率 (0-1)：衡量生成文本中有多少內容是相關的。理想值 > 0.7'
    ),
    'bert_score_r': fields.Float(
        description='BERTScore 召回率 (0-1)：衡量參考文本中有多少內容被覆蓋到。理想值 > 0.7'
    ),
    'bert_score_f1': fields.Float(
        description='BERTScore F1 分數 (0-1)：精確率和召回率的調和平均。理想值 > 0.7'
    ),
    'semantic_similarity': fields.Float(
        description='語義相似度 (0-1)：使用中文 BERT 計算的語義層面相似程度。理想值 > 0.8'
    ),
    'rouge_score': fields.Float(
        description='ROUGE-L 分數 (0-1)：評估生成文本與參考文本的重疊程度。理想值 > 0.4'
    ),
    'mrr_score': fields.Float(
        description='平均倒數排名分數 (0-1)：評估檢索結果的排序質量。理想值 > 0.5'
    ),
    'exact_match': fields.Float(
        description='完全匹配分數 (0-1)：計算生成文本和參考文本的詞彙重疊程度。理想值 > 0.3'
    )
})

response_model = api.model('Response', {
    'model_response': fields.Nested(api.model('ModelResponse', {
        'answer': fields.String(description='模型生成的回答內容'),
        'metrics': fields.Nested(metrics_model)
    })),
    'retrieved_docs': fields.List(
        fields.String, 
        description='檢索到的相關文檔列表'
    ),
    'similarity_scores': fields.List(
        fields.Float, 
        description='每個檢索文檔與查詢的相似度分數 (0-1)'
    ),
    'execution_time': fields.Float(
        description='API 執行時間（秒）'
    )
})

# 定義允許的模型類型
ALLOWED_MODELS = ["gpt-4", "gemini"]

# 定義每種模型類型支援的具體模型版本
SUPPORTED_MODEL_NAMES = {
    "gpt-4": ["gpt-4", "gpt-4-32k", "gpt-4-turbo"],
    "gemini": ["gemini-1.0-pro", "gemini-1.0-ultra", "gemini-2.0-flash", "gemini-2.0-pro"]
}

def fetch_documents_from_api():
    """從 API 獲取文檔數據"""
    url = NOCODB_API_URL
    headers = {
        'accept': 'application/json',
        'xc-token': NOCODB_API_KEY
    }
    params = {
        'limit': 25,
        'shuffle': 0,
        'offset': 0
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        # 從 API 回傳的資料中提取有用的文本信息
        documents = []
        for record in data.get('list', []):
            # 組合各個欄位的資訊成為有意義的文本
            doc_text = f"人格類型：{record.get('人格類型', '')}，"
            doc_text += f"類型名稱：{record.get('類型名稱', '')}，"
            doc_text += f"個性特點：{record.get('個性特點', '')}，"
            doc_text += f"代表人物：{record.get('代表人物', '')}，"
            doc_text += f"滿通風格：{record.get('滿通風格', '')}，"
            doc_text += f"優勢：{record.get('優勢', '')}，"
            doc_text += f"弱勢：{record.get('弱勢', '')}，"
            doc_text += f"愛情觀：{record.get('愛情觀', '')}"
            
            documents.append(doc_text)
            
        return documents
    except Exception as e:
        print(f"從 API 獲取文檔時出錯: {str(e)}")
        return []

# 在應用啟動時從 API 獲取文檔
documents = fetch_documents_from_api()

# 如果 API 獲取失敗，使用備用文檔
if not documents:
    documents = [
        "這是第一個示例文檔，用於測試檢索系統。",
        "第二個文檔包含一些不同的內容。",
        "第三個文檔有其他獨特的信息。",
        "這是第四個文檔，用於展示檢索效果。",
        "第五個文檔包含更多的測試內容。"
    ]

class Retriever:
    def __init__(self, documents: List[str], model_name: str):
        self.documents = documents
        self.model_name = model_name
        self.model = None
        self.doc_embeddings = None
        self.bm25 = None
        self._initialize_models()
    
    def _initialize_models(self):
        # 初始化 Sentence Transformer
        try:
            logging.info(f"開始載入模型: {self.model_name}")
            logging.info(f"當前工作目錄: {os.getcwd()}")
            logging.info(f"模型快取目錄: {os.getenv('TRANSFORMERS_CACHE', '未設定')}")
            
            self.model = SentenceTransformer(self.model_name)
            logging.info(f"模型載入成功: {self.model_name}")
            
            if self.documents:
                logging.info(f"開始編碼 {len(self.documents)} 個文件")
                self.doc_embeddings = self.model.encode(self.documents, convert_to_tensor=False, show_progress_bar=False)
                logging.info("文件編碼完成")
        except Exception as e:
            logging.error(f"載入主要模型時發生錯誤: {str(e)}")
            logging.error(f"錯誤類型: {type(e).__name__}")
            logging.error(f"錯誤堆疊: {traceback.format_exc()}")
            
            try:
                backup_model = 'all-MiniLM-L6-v2'
                logging.info(f"嘗試使用備用模型: {backup_model}")
                self.model = SentenceTransformer(backup_model)
                logging.info(f"備用模型載入成功: {backup_model}")
                
                if self.documents:
                    logging.info(f"使用備用模型編碼 {len(self.documents)} 個文件")
                    self.doc_embeddings = self.model.encode(self.documents, convert_to_tensor=False, show_progress_bar=False)
                    logging.info("備用模型文件編碼完成")
            except Exception as e2:
                logging.error(f"載入備用模型時也發生錯誤: {str(e2)}")
                logging.error(f"備用模型錯誤類型: {type(e2).__name__}")
                logging.error(f"備用模型錯誤堆疊: {traceback.format_exc()}")
                raise RuntimeError("無法載入任何可用的模型") from e2
        
        # 初始化 BM25
        try:
            tokenized_documents = []
            for doc in self.documents:
                if doc and isinstance(doc, str):
                    tokens = nltk.word_tokenize(doc.strip())
                    tokenized_documents.append(tokens if tokens else ["placeholder"])
                else:
                    tokenized_documents.append(["placeholder"])
            self.bm25 = BM25Okapi(tokenized_documents)
            logging.info("BM25 初始化成功")
        except Exception as e:
            logging.error(f"BM25 初始化錯誤: {str(e)}")
            logging.error(f"BM25 錯誤堆疊: {traceback.format_exc()}")
            self.bm25 = None
    
    def top_k_retrieval(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """使用餘弦相似度進行 top-k 檢索"""
        query_embedding = self.model.encode([query], convert_to_tensor=False, show_progress_bar=False)[0]
        similarities = cosine_similarity([query_embedding], self.doc_embeddings)[0]
        
        # 獲取 top-k 的索引和分數
        top_k_indices = similarities.argsort()[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            # 格式化相似度分數為小數點後兩位
            score = round(float(similarities[idx]), 2)
            doc_text = f"{self.documents[idx]} (相似度: {score})"
            results.append({
                "document": doc_text,
                "score": score
            })
        
        return results
    
    def mmr_retrieval(self, query: str, k: int = 3, lambda_param: float = 0.5) -> List[Dict[str, Any]]:
        try:
            # 使用較簡單的編碼方式
            query_embedding = self.model.encode([query], convert_to_tensor=False)[0]
            doc_scores = cosine_similarity([query_embedding], self.doc_embeddings)[0]
            
            selected_indices = []
            unselected_indices = list(range(len(self.documents)))
            
            for _ in range(k):
                if not unselected_indices:
                    break
                    
                mmr_scores = []
                for idx in unselected_indices:
                    if not selected_indices:
                        mmr_score = doc_scores[idx]
                    else:
                        selected_embeddings = self.doc_embeddings[selected_indices]
                        sim_to_selected = cosine_similarity([self.doc_embeddings[idx]], selected_embeddings)[0].max()
                        mmr_score = lambda_param * doc_scores[idx] - (1 - lambda_param) * sim_to_selected
                    
                    mmr_scores.append((idx, mmr_score))
                
                best_idx, _ = max(mmr_scores, key=lambda x: x[1])
                selected_indices.append(best_idx)
                unselected_indices.remove(best_idx)
            
            results = []
            for idx in selected_indices:
                results.append({
                    'document': self.documents[idx],
                    'score': round(float(doc_scores[idx]), 2)
                })
            return results
        except Exception as e:
            print(f"MMR 檢索過程中發生錯誤: {str(e)}")
            return []

    def process_query(self, query: str, k: int = 3, use_mmr: bool = False) -> Dict[str, Any]:
        """處理查詢並返回結果"""
        start_time = time.time()
        
        try:
            if use_mmr:
                results = self.mmr_retrieval(query, k)
            else:
                results = self.top_k_retrieval(query, k)
            
            execution_time = round(time.time() - start_time, 2)
            
            return {
                "query": query,
                "results": results,
                "execution_time": execution_time
            }
            
        except Exception as e:
            logging.error(f"處理查詢時發生錯誤: {str(e)}")
            return {
                "query": query,
                "results": [],
                "execution_time": round(time.time() - start_time, 2),
                "error": str(e)
            }

def get_gpt_response(messages: List[Dict[str, str]], model_name: str = None) -> str:
    try:
        response = openai.ChatCompletion.create(
            model=model_name if model_name else DEFAULT_GPT_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT API 錯誤: {str(e)}")
        return ""

def get_gemini_response(prompt: str, model_name: str = None) -> str:
    try:
        # 使用指定的模型名稱或默認值
        model = genai.GenerativeModel(model_name if model_name else os.getenv('GEMINI_MODEL', 'gemini-2.0-flash'))
        chat = model.start_chat(history=[])
        
        system_prompt = """作為專業的 AI 助手，你必須嚴格遵循以下指南：

1. 內容分析：
   - 識別參考資料中的所有關鍵詞和核心概念
   - 確保回答包含這些關鍵詞
   - 維持與參考資料相同的專業術語使用

2. 回答要求：
   - 直接使用參考資料中的原始表述
   - 保持與參考資料一致的描述方式
   - 確保每個關鍵概念都有對應的原文支持

3. 準確性控制：
   - 不要改變關鍵詞的原始含義
   - 優先使用直接引用而非改寫
   - 確保回答與參考資料的描述完全匹配
   - 禁止添加參考資料以外的內容

4. 格式規範：
   - 回答必須完全基於參考資料
   - 保留原始關鍵詞的使用方式
   - 維持專業且精確的表達"""
        
        full_prompt = f"{system_prompt}\n\n{prompt}"
        response = chat.send_message(full_prompt, generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=1000
        ))
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API 錯誤: {str(e)}")
        return ""

def evaluate_model_quality(reference: str, model_response: str) -> float:
    """評估模型回應的質量"""
    if not model_response:
        return 0.0
        
    # 使用 ROUGE-L 作為質量評估指標
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(reference, model_response)
    return round(score['rougeL'].fmeasure, 2)

def calculate_recall_at_k(retrieved_docs: List[Dict[str, Any]], relevant_docs: List[str]) -> float:
    """計算 Recall@k"""
    if not relevant_docs:
        return 0.0
    
    retrieved_set = set(doc['document'] for doc in retrieved_docs)
    relevant_set = set(relevant_docs)
    
    recall = len(retrieved_set.intersection(relevant_set)) / len(relevant_set)
    return round(recall, 2)

def calculate_metrics(retrieved_docs: List[Dict[str, Any]], query: str, model: str = DEFAULT_GPT_MODEL) -> Dict[str, float]:
    # 計算 BLEU
    references = [[doc['document']] for doc in retrieved_docs]
    hypothesis = [query]
    bleu = round(sacrebleu.corpus_bleu(hypothesis, references).score / 100, 2)

    # 計算 ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(query, retrieved_docs[0]['document'])

    # 計算 Recall@k
    recall_at_k = calculate_recall_at_k(retrieved_docs, [doc['document'] for doc in retrieved_docs])
    
    # 根據指定的模型獲取回應
    context = "\n".join([doc['document'] for doc in retrieved_docs])
    prompt = f"基於以下參考資料回答問題。\n\n參考資料：{context}\n\n問題：{query}"
    
    # 初始化質量分數
    gpt_4_quality = 0.0
    gemini_quality = 0.0
    
    # 系統提示詞
    system_prompt = """作為專業的 AI 助手，你必須嚴格遵循以下指南：

1. 內容分析：
   - 識別參考資料中的所有關鍵詞和核心概念
   - 確保回答包含這些關鍵詞
   - 維持與參考資料相同的專業術語使用

2. 回答要求：
   - 直接使用參考資料中的原始表述
   - 保持與參考資料一致的描述方式
   - 確保每個關鍵概念都有對應的原文支持

3. 準確性控制：
   - 不要改變關鍵詞的原始含義
   - 優先使用直接引用而非改寫
   - 確保回答與參考資料的描述完全匹配
   - 禁止添加參考資料以外的內容

4. 格式規範：
   - 回答必須完全基於參考資料
   - 保留原始關鍵詞的使用方式
   - 維持專業且精確的表達"""
    
    # 根據模型選擇評估
    if model == "gemini":
        gemini_response = get_gemini_response(prompt, model_name=model_name)
        if gemini_response:
            gemini_quality = evaluate_model_quality(retrieved_docs[0]['document'], gemini_response)
    else:  # 預設使用 gpt-4
        gpt_4_response = get_gpt_response(prompt, model_name=model_name)
        if gpt_4_response:
            gpt_4_quality = evaluate_model_quality(retrieved_docs[0]['document'], gpt_4_response)

    return {
        'bleu': bleu,
        'rouge1': round(rouge_scores['rouge1'].fmeasure, 2),
        'rouge2': round(rouge_scores['rouge2'].fmeasure, 2),
        'rougeL': round(rouge_scores['rougeL'].fmeasure, 2),
        'mrr': round(1.0 / (1 + np.argmax([doc['score'] for doc in retrieved_docs])), 2),
        'recall_at_k': recall_at_k,
        'gpt_4_quality': gpt_4_quality,
        'gemini_quality': gemini_quality
    }

def calculate_semantic_similarity(text1: str, text2: str, model_name: str = 'ckiplab/bert-base-chinese') -> float:
    """計算兩段文本的語義相似度"""
    try:
        # 載入模型和分詞器
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # 對文本進行編碼
        inputs1 = tokenizer(text1, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs2 = tokenizer(text2, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        # 獲取文本嵌入
        with torch.no_grad():
            embeddings1 = model(**inputs1).last_hidden_state.mean(dim=1)
            embeddings2 = model(**inputs2).last_hidden_state.mean(dim=1)
        
        # 計算餘弦相似度
        similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
        return round(float(similarity[0]), 2)
    except Exception as e:
        logging.error(f"計算語義相似度時發生錯誤: {str(e)}")
        return 0.0

def calculate_exact_match(prediction: str, reference: str) -> float:
    """計算完全匹配分數"""
    # 對文本進行預處理：分詞並去除標點符號
    prediction_tokens = set(jieba.cut(prediction))
    reference_tokens = set(jieba.cut(reference))
    
    # 計算交集比例
    if not reference_tokens:
        return 0.0
    
    intersection = prediction_tokens.intersection(reference_tokens)
    return round(len(intersection) / len(reference_tokens), 2)

def get_model_response(query: str, model: str, context: str, model_name: str = None, custom_prompt: str = None) -> Dict[str, Any]:
    """根據選擇的模型生成回應並計算指標"""
    # 使用自定義系統提示詞或默認系統提示詞
    system_prompt = custom_prompt if custom_prompt else """作為專業的 AI 助手，你必須嚴格遵循以下指南：

1. 內容分析：
   - 識別參考資料中的所有關鍵詞和核心概念
   - 確保回答包含這些關鍵詞
   - 維持與參考資料相同的專業術語使用

2. 回答要求：
   - 直接使用參考資料中的原始表述
   - 保持與參考資料一致的描述方式
   - 確保每個關鍵概念都有對應的原文支持

3. 準確性控制：
   - 不要改變關鍵詞的原始含義
   - 優先使用直接引用而非改寫
   - 確保回答與參考資料的描述完全匹配
   - 禁止添加參考資料以外的內容

4. 格式規範：
   - 回答必須完全基於參考資料
   - 保留原始關鍵詞的使用方式
   - 維持專業且精確的表達"""

    # 固定的用戶提示詞格式
    user_prompt = f"參考資料：{context}\n\n問題：{query}"
    
    try:
        if model == "gemini":
            # 使用指定的 Gemini 模型版本（如果提供）
            gemini_model_name = model_name if model_name else os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')
            # 組合系統提示詞和用戶提示詞
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = get_gemini_response(full_prompt, model_name=gemini_model_name)
            quality_score = evaluate_model_quality(context, response) if response else 0.0
            
            # 計算 BERTScore
            P, R, F1 = bert_score([response], [context], lang='zh', verbose=False)
            bert_scores = {
                'precision': round(float(P[0]), 2),
                'recall': round(float(R[0]), 2),
                'f1': round(float(F1[0]), 2)
            }
            
            # 計算語義相似度
            semantic_sim = calculate_semantic_similarity(response, context)
            
            # 計算完全匹配分數
            exact_match_score = calculate_exact_match(response, context)
            
            # 計算 ROUGE 分數
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(response, context) if response else None
            
            return {
                'answer': response,
                'metrics': {
                    'bert_score_p': bert_scores['precision'],
                    'bert_score_r': bert_scores['recall'],
                    'bert_score_f1': bert_scores['f1'],
                    'semantic_similarity': semantic_sim,
                    'rouge_score': round(rouge_scores['rougeL'].fmeasure, 2) if rouge_scores else 0.0,
                    'mrr_score': quality_score,
                    'exact_match': exact_match_score
                }
            }
        else:  # GPT-4
            # 使用指定的 GPT 模型版本（如果提供）
            gpt_model_name = model_name if model_name else DEFAULT_GPT_MODEL
            # 使用 messages 陣列分別傳遞系統提示詞和用戶提示詞
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = get_gpt_response(messages, model_name=gpt_model_name)
            quality_score = evaluate_model_quality(context, response) if response else 0.0
            
            # 計算 BERTScore
            P, R, F1 = bert_score([response], [context], lang='zh', verbose=False)
            bert_scores = {
                'precision': round(float(P[0]), 2),
                'recall': round(float(R[0]), 2),
                'f1': round(float(F1[0]), 2)
            }
            
            # 計算語義相似度
            semantic_sim = calculate_semantic_similarity(response, context)
            
            # 計算完全匹配分數
            exact_match_score = calculate_exact_match(response, context)
            
            # 計算 ROUGE 分數
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(response, context) if response else None
            
            return {
                'answer': response,
                'metrics': {
                    'bert_score_p': bert_scores['precision'],
                    'bert_score_r': bert_scores['recall'],
                    'bert_score_f1': bert_scores['f1'],
                    'semantic_similarity': semantic_sim,
                    'rouge_score': round(rouge_scores['rougeL'].fmeasure, 2) if rouge_scores else 0.0,
                    'mrr_score': quality_score,
                    'exact_match': exact_match_score
                }
            }
    except Exception as e:
        logging.error(f"生成模型回應時發生錯誤: {str(e)}")
        return {
            'answer': str(e),
            'metrics': {
                'bert_score_p': 0.0,
                'bert_score_r': 0.0,
                'bert_score_f1': 0.0,
                'semantic_similarity': 0.0,
                'rouge_score': 0.0,
                'mrr_score': 0.0,
                'exact_match': 0.0
            }
        }

# 在定義完 Retriever 類之後初始化全局檢索器實例
retriever = Retriever(documents=documents, model_name=DEFAULT_EMBEDDING_MODEL)

@ns.route('/retrieve')
class Retrieve(Resource):
    @ns.expect(retrieve_request)
    @ns.marshal_with(response_model)
    @ns.doc(
        description='根據查詢檢索相關文檔並生成回答，同時計算多個評估指標\n\n'
                   '使用建議：\n'
                   '1. 一般查詢建議使用預設參數\n'
                   '2. 需要更多相關文檔時可調整 k_value\n'
                   '3. 需要更多樣化的檢索結果時使用 mmr 方法\n'
                   '4. 對回答質量要求較高時建議使用 GPT-4\n'
                   '5. 需要更快速的回應時可使用 Gemini\n'
                   '6. 溫度參數建議保持在 0.2-0.5 之間'
    )
    def post(self):
        start_time = time.time()
        data = api.payload
        query = data['query']
        embedding_model = data.get('embedding_model', DEFAULT_EMBEDDING_MODEL)
        retriever_method = data.get('retriever_method', DEFAULT_RETRIEVER_METHOD)
        k = data.get('k_value', DEFAULT_K_VALUE)
        model = data.get('model', DEFAULT_GPT_MODEL)
        model_name = data.get('model_name')
        temperature = data.get('temperature', 0.2)
        custom_prompt = data.get('prompt')  # 獲取自定義 prompt

        try:
            # 驗證模型類型
            if model not in ALLOWED_MODELS:
                raise ValueError(f"不支援的模型類型: {model}。支援的模型類型有: {', '.join(ALLOWED_MODELS)}")

            # 驗證具體模型版本（如果提供）
            if model_name:
                if model_name not in SUPPORTED_MODEL_NAMES[model]:
                    raise ValueError(f"不支援的模型版本: {model_name}。{model} 支援的版本有: {', '.join(SUPPORTED_MODEL_NAMES[model])}")

            # 驗證檢索方法
            if retriever_method not in ['top_k', 'mmr']:
                raise ValueError(f"不支援的檢索方法: {retriever_method}。支援的方法有: top_k, mmr")

            # 使用檢索器獲取相關文檔
            if retriever_method == 'mmr':
                docs = retriever.mmr_retrieval(query=query, k=k)
            else:
                docs = retriever.top_k_retrieval(query=query, k=k)

            # 使用檢索到的文檔作為上下文生成回應
            context = "\n".join([doc['document'] for doc in docs])
            
            # 根據選擇的模型和具體版本生成回應，並傳入自定義 prompt
            model_response = get_model_response(
                query=query,
                model=model,
                context=context,
                model_name=model_name if model_name else None,
                custom_prompt=custom_prompt
            )

            # 格式化執行時間為小數點後兩位
            execution_time = round(time.time() - start_time, 2)

            return {
                'model_response': model_response,
                'retrieved_docs': [doc['document'] for doc in docs],
                'similarity_scores': [round(doc['score'], 2) for doc in docs],
                'execution_time': execution_time
            }

        except ValueError as ve:
            logging.error(f"參數驗證錯誤: {str(ve)}")
            return {
                'model_response': {
                    'answer': str(ve),
                    'metrics': {'bleu_score': 0.0, 'rouge_score': 0.0, 'mrr_score': 0.0}
                },
                'retrieved_docs': [],
                'similarity_scores': [],
                'execution_time': round(time.time() - start_time, 2)
            }, 400

if __name__ == '__main__':
     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000))) 