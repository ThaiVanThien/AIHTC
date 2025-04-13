# Dữ liệu huấn luyện cho mô hình NLP

Thư mục này chứa các tệp dữ liệu huấn luyện cho mô hình NLP.

## Định dạng dữ liệu

Hệ thống hỗ trợ các định dạng dữ liệu huấn luyện sau:

### 1. JSON

Tệp JSON phải có cấu trúc là một mảng các đối tượng, mỗi đối tượng đại diện cho một mẫu dữ liệu với các trường bắt buộc:

```json
[
  {
    "question": "Kế toán là gì?",
    "context": "Kế toán là một hệ thống thông tin và đo lường các hoạt động kinh tế của một tổ chức.",
    "answer": "Kế toán là một hệ thống thông tin và đo lường các hoạt động kinh tế của một tổ chức."
  },
  {
    "question": "Phương trình kế toán cơ bản là gì?",
    "context": "Phương trình kế toán cơ bản là: Tài sản = Nợ phải trả + Vốn chủ sở hữu.",
    "answer": "Phương trình kế toán cơ bản là: Tài sản = Nợ phải trả + Vốn chủ sở hữu."
  }
]
```

### 2. CSV

Tệp CSV phải có các cột tương ứng với ba trường bắt buộc: `question`, `context` và `answer`.

```
question,context,answer
"Kế toán là gì?","Kế toán là một hệ thống thông tin và đo lường các hoạt động kinh tế của một tổ chức.","Kế toán là một hệ thống thông tin và đo lường các hoạt động kinh tế của một tổ chức."
"Phương trình kế toán cơ bản là gì?","Phương trình kế toán cơ bản là: Tài sản = Nợ phải trả + Vốn chủ sở hữu.","Phương trình kế toán cơ bản là: Tài sản = Nợ phải trả + Vốn chủ sở hữu."
```

### 3. Excel (XLSX/XLS)

Tệp Excel phải có một bảng tính với ba cột tương ứng với ba trường bắt buộc: `question`, `context` và `answer`.

## Cách tải lên dữ liệu huấn luyện

Có hai cách để tải dữ liệu huấn luyện:

### 1. Sử dụng API

```bash
# Tải lên tệp JSON
curl -X POST "http://localhost:8002/api/v1/nlp/upload-training-file" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@đường_dẫn_đến_tệp.json;type=application/json" \
  -F "file_type=json"

# Tải lên tệp CSV
curl -X POST "http://localhost:8002/api/v1/nlp/upload-training-file" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@đường_dẫn_đến_tệp.csv;type=text/csv" \
  -F "file_type=csv"

# Tải lên tệp Excel
curl -X POST "http://localhost:8002/api/v1/nlp/upload-training-file" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@đường_dẫn_đến_tệp.xlsx;type=application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" \
  -F "file_type=xlsx"
```

### 2. Trực tiếp đặt tệp vào thư mục

Sao chép tệp dữ liệu huấn luyện trực tiếp vào thư mục `data/training/`. Hệ thống sẽ tự động phát hiện và sử dụng các tệp trong thư mục này khi bắt đầu quá trình huấn luyện. 