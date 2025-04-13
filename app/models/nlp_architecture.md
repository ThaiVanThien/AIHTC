# Kiến trúc hệ thống QA (Question Answering)

## Giới thiệu

Hệ thống QA sử dụng mô hình vi-mrc để trả lời câu hỏi dựa trên ngữ cảnh. Đây là mô hình đã được huấn luyện trên dữ liệu tiếng Việt và có khả năng hiểu câu hỏi, phân tích ngữ cảnh, và trích xuất câu trả lời chính xác.

## Luồng xử lý

```
  Câu hỏi + Ngữ cảnh
         │
         ▼
┌──────────────────┐
│  Tokenizer        │
└──────────────────┘
         │
         ▼
┌──────────────────┐
│  vi-mrc Model    │
└──────────────────┘
         │
         ▼
┌──────────────────┐
│ Xác định vị trí  │
│  câu trả lời     │
└──────────────────┘
         │
         ▼
┌──────────────────┐
│ Trích xuất câu   │
│    trả lời       │
└──────────────────┘
         │
         ▼
     Câu trả lời
```

## Thành phần chính

### Mô hình vi-mrc
- Mô hình dựa trên kiến trúc BERT được huấn luyện đặc biệt cho tiếng Việt
- Fine-tuned trên dữ liệu câu hỏi - câu trả lời tiếng Việt
- Hỗ trợ các loại câu hỏi: Ai, Cái gì, Ở đâu, Khi nào, Bao nhiêu, Tại sao...

### Cách hoạt động
1. **Tokenization**: Chuyển đổi câu hỏi và ngữ cảnh thành các token
2. **Encoding**: Mã hóa các token thành vector đầu vào
3. **Prediction**: Dự đoán vị trí bắt đầu và kết thúc của câu trả lời trong ngữ cảnh
4. **Extraction**: Trích xuất câu trả lời dựa trên vị trí dự đoán

### API Endpoint
- **POST /api/v1/nlp/answer-question**
  - Input: 
    - `question`: Câu hỏi cần trả lời
    - `context`: Ngữ cảnh chứa câu trả lời
  - Output:
    - `answer`: Câu trả lời được trích xuất
    - `confidence`: Độ tin cậy của câu trả lời
    - `start`: Vị trí bắt đầu trong ngữ cảnh
    - `end`: Vị trí kết thúc trong ngữ cảnh

## Ví dụ
### Input
- **Question**: "Doanh thu quý 1 là bao nhiêu?"
- **Context**: "Doanh thu quý 1 là 500 tỷ đồng, tăng 20% so với cùng kỳ năm ngoái."

### Output
```json
{
  "answer": "500 tỷ đồng",
  "confidence": 0.95,
  "start": 18,
  "end": 30
}
```

## Cải tiến trong tương lai
- Tích hợp với hệ thống phân loại ý định (Intent Detection)
- Hỗ trợ trả lời câu hỏi từ nhiều nguồn ngữ cảnh
- Cải thiện khả năng xử lý câu hỏi phức tạp và trừu tượng 