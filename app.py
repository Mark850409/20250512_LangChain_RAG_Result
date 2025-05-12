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

# 載入環境變數
load_dotenv()

app = Flask(__name__)
api = Api(app, version='1.0', 
          title='向量檢索 API',
          description='支援多種檢索方法和評估指標的向量檢索 API',
          doc='/docs')

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

# API 請求和回應模型定義
retrieve_request = api.model('RetrieveRequest', {
    'embedding_model': fields.String(description='使用的嵌入模型名稱', required=False, default=DEFAULT_EMBEDDING_MODEL),
    'retriever_method': fields.String(description='檢索方法 (top_k 或 mmr)', required=False, default=DEFAULT_RETRIEVER_METHOD),
    'k_value': fields.Integer(description='返回的文檔數量', required=False, default=DEFAULT_K_VALUE),
    'query': fields.String(description='查詢文本', required=True),
    'model': fields.String(description='使用的生成模型 (gpt-4 或 gemini)', required=False, default=DEFAULT_GPT_MODEL)
})

document_model = api.model('Document', {
    'document': fields.String(description='文檔內容'),
    'score': fields.Float(description='相似度分數')
})

metrics_model = api.model('Metrics', {
    'bleu': fields.Float(description='BLEU 分數'),
    'rouge1': fields.Float(description='ROUGE-1 分數'),
    'rouge2': fields.Float(description='ROUGE-2 分數'),
    'rougeL': fields.Float(description='ROUGE-L 分數'),
    'mrr': fields.Float(description='MRR 分數'),
    'recall_at_k': fields.Float(description='Recall@k 分數'),
    'gpt_4_quality': fields.Float(description='GPT-4 回答質量評分'),
    'gemini_quality': fields.Float(description='Gemini 2.0 回答質量評分')
})

response_model = api.model('Response', {
    'model_response': fields.Nested(api.model('ModelResponse', {
        'answer': fields.String(description='Model response'),
        'metrics': fields.Nested(api.model('Metrics', {
            'bleu_score': fields.Float(description='BLEU score'),
            'rouge_score': fields.Float(description='ROUGE score'),
            'mrr_score': fields.Float(description='MRR score')
        }))
    })),
    'retrieved_docs': fields.List(fields.String, description='Retrieved documents'),
    'similarity_scores': fields.List(fields.Float, description='Similarity scores'),
    'execution_time': fields.Float(description='API execution time in seconds')
})

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

def get_gpt_response(prompt: str) -> str:
    try:
        messages = [
            {"role": "system", "content": """你是一個專業的 AI 助手，專門提供高度相關且精確的回答。請嚴格遵循以下指南：

1. 分析要求：
   - 仔細識別參考資料中的關鍵詞和重要概念
   - 確保回答中包含這些關鍵詞
   - 保持與參考資料的用詞一致性

2. 回答結構：
   - 使用參考資料中的原始措辭和表達方式
   - 按照參考資料的邏輯順序組織回答
   - 確保每個重要概念都有對應的關鍵詞支持

3. 品質控制：
   - 不要改寫或意譯關鍵術語
   - 直接引用參考資料中的重要片段
   - 確保回答與參考資料的描述完全一致
   - 避免添加未在參考資料中出現的解釋或推論

4. 輸出要求：
   - 回答必須完全基於提供的參考資料
   - 使用參考資料中的原始關鍵詞
   - 保持專業且準確的表達方式"""},
            {"role": "user", "content": prompt}
        ]
        
        response = openai.ChatCompletion.create(
            model=DEFAULT_GPT_MODEL,
            messages=messages,
            temperature=0.2,  # 降低溫度以提高準確性和一致性
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT API 錯誤: {str(e)}")
        return ""

def get_gemini_response(prompt: str) -> str:
    try:
        model = genai.GenerativeModel(os.getenv('GEMINI_MODEL', 'gemini-2.0-flash'))
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
            temperature=0.2,  # 降低溫度以提高準確性和一致性
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
        gemini_response = get_gemini_response(prompt)
        if gemini_response:
            gemini_quality = evaluate_model_quality(retrieved_docs[0]['document'], gemini_response)
    else:  # 預設使用 gpt-4
        gpt_4_response = get_gpt_response(prompt)
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

def get_model_response(query: str, model: str, context: str) -> Dict[str, Any]:
    """根據選擇的模型生成回應並計算指標"""
    prompt = f"""基於以下參考資料回答問題。請遵循以下要求：
1. 提供一個完整且連貫的回答
2. 使用參考資料中的關鍵詞和表述
3. 保持專業且精確的表達
4. 以摘要形式組織內容

參考資料：{context}

問題：{query}"""
    
    try:
        if model == "gemini":
            response = get_gemini_response(prompt)
            quality_score = evaluate_model_quality(context, response) if response else 0.0
            
            # 計算其他指標
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(response, context) if response else None
            
            return {
                'answer': response,
                'metrics': {
                    'bleu_score': round(sacrebleu.corpus_bleu([response], [[context]]).score / 100, 2) if response else 0.0,
                    'rouge_score': round(rouge_scores['rougeL'].fmeasure, 2) if rouge_scores else 0.0,
                    'mrr_score': quality_score
                }
            }
        else:  # 預設使用 gpt-4
            response = get_gpt_response(prompt)
            quality_score = evaluate_model_quality(context, response) if response else 0.0
            
            # 計算其他指標
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(response, context) if response else None
            
            return {
                'answer': response,
                'metrics': {
                    'bleu_score': round(sacrebleu.corpus_bleu([response], [[context]]).score / 100, 2) if response else 0.0,
                    'rouge_score': round(rouge_scores['rougeL'].fmeasure, 2) if rouge_scores else 0.0,
                    'mrr_score': quality_score
                }
            }
    except Exception as e:
        logging.error(f"生成模型回應時發生錯誤: {str(e)}")
        return {
            'answer': "",
            'metrics': {
                'bleu_score': 0.0,
                'rouge_score': 0.0,
                'mrr_score': 0.0
            }
        }

@ns.route('/retrieve')
class DocumentRetrieval(Resource):
    @ns.expect(retrieve_request)
    @ns.marshal_with(response_model)
    @ns.doc(description='根據查詢檢索相關文檔並計算評估指標')
    def post(self):
        """
        檢索文檔並計算相關指標
        
        從文檔集合中檢索與查詢最相關的文檔，並計算相關性指標。
        支援 top-k 和 MMR 兩種檢索方法，以及 GPT-4 和 Gemini 2.0 兩種生成模型。
        """
        start_time = time.time()
        data = api.payload
        query = data['query']
        model_name = data.get('embedding_model', DEFAULT_EMBEDDING_MODEL)
        method = data.get('retriever_method', DEFAULT_RETRIEVER_METHOD)
        k = data.get('k_value', DEFAULT_K_VALUE)
        model = data.get('model', DEFAULT_GPT_MODEL)

        try:
            # 初始化檢索器
            retriever = Retriever(documents, model_name)

            # 根據選擇的方法進行檢索
            if method == 'mmr':
                retrieved_docs = retriever.mmr_retrieval(query, k)
            else:
                retrieved_docs = retriever.top_k_retrieval(query, k)

            # 使用檢索到的文檔作為上下文生成回應
            context = "\n".join([doc['document'] for doc in retrieved_docs])
            
            # 初始化回應變數
            gpt4_response = None
            gemini_response = None
            
            # 只調用使用者選擇的模型
            if model == "gpt-4":
                gpt4_response = get_model_response(query, "gpt-4", context)
            elif model == "gemini":
                gemini_response = get_model_response(query, "gemini", context)

            # 格式化執行時間為小數點後兩位
            execution_time = round(time.time() - start_time, 2)

            # 建立回應字典，只包含使用者選擇的模型回應
            model_response = {}
            
            # 根據使用者選擇的模型返回對應的回應
            if model == "gpt-4" and gpt4_response and gpt4_response['answer']:
                model_response = {
                    'answer': gpt4_response['answer'],
                    'metrics': gpt4_response['metrics']
                }
            elif model == "gemini" and gemini_response and gemini_response['answer']:
                model_response = {
                    'answer': gemini_response['answer'],
                    'metrics': gemini_response['metrics']
                }

            return {
                'model_response': model_response,
                'retrieved_docs': [doc['document'] for doc in retrieved_docs],
                'similarity_scores': [round(doc['score'], 2) for doc in retrieved_docs],
                'execution_time': execution_time
            }
            
        except Exception as e:
            logging.error(f"處理請求時發生錯誤: {str(e)}")
            logging.error(traceback.format_exc())
            # 根據使用者選擇的模型返回空結構
            empty_response = {}
            if model == "gpt-4":
                empty_response = {'answer': '', 'metrics': {'bleu_score': 0.0, 'rouge_score': 0.0, 'mrr_score': 0.0}}
            elif model == "gemini":
                empty_response = {'answer': '', 'metrics': {'bleu_score': 0.0, 'rouge_score': 0.0, 'mrr_score': 0.0}}
                
            return {
                'model_response': empty_response,
                'retrieved_docs': [],
                'similarity_scores': [],
                'execution_time': round(time.time() - start_time, 2)
            }

if __name__ == '__main__':
     app.run(host="0.0.0.0", port=5000, debug=True)