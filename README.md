# 向量檢索 API

這是一個使用 Flask 建立的簡單向量檢索 API，支援多種檢索方法和評估指標。

## 安裝需求

1. 安裝所需套件：
```bash
pip install -r requirements.txt
```

## 啟動服務

```bash
python app.py
```

服務將在 http://localhost:5000 啟動。

## API 使用說明

### 檢索端點：POST /retrieve

請求參數（JSON）：

```json
{
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",  // 選用的嵌入模型
    "retriever_method": "top_k",  // 檢索方法：'top_k' 或 'mmr'
    "k_value": 3,  // 要檢索的文檔數量
    "query": "查詢文本"  // 必填：查詢內容
}
```

回應格式：

```json
{
    "retrieved_documents": [
        {
            "document": "檢索到的文檔內容",
            "score": 0.85  // 相似度分數
        }
    ],
    "generated_response": "生成的回應文本",
    "metrics": {
        "bleu": 0.5,
        "rouge1": 0.6,
        "rouge2": 0.4,
        "rougeL": 0.5,
        "mrr": 1.0
    }
}
```

## 支援的功能

1. 檢索方法：
   - Top-K：基於餘弦相似度的傳統檢索
   - MMR (Maximal Marginal Relevance)：平衡相關性和多樣性的檢索

2. 評估指標：
   - BLEU：評估生成文本的品質
   - ROUGE：評估文本相似度
   - MRR：評估檢索排序品質

## 示例

```bash
curl -X POST http://localhost:5000/retrieve \
     -H "Content-Type: application/json" \
     -d '{
           "query": "什麼是機器學習？",
           "k_value": 3,
           "retriever_method": "mmr"
         }'
``` 