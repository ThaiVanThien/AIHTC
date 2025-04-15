import json
import logging
import os
from typing import Dict, List, Optional, Any

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentRouter:
    def __init__(self, default_provider: str = "gemini"):
        """
        Khởi tạo IntentRouter với nhà cung cấp AI mặc định
        """
        self.default_provider = default_provider
        logger.info(f"Khởi tạo IntentRouter với provider mặc định: {self.default_provider}")
        
        # Mô phỏng service
        self.gemini_service = GeminiService()
        self.openai_service = OpenAIService()
        self.vimrc_service = VIMRCService()
        self.document_store = DocumentStore()
    
    def analyze_query(self, query: str, provider: str = None) -> Dict[str, Any]:
        """
        Phân tích ý định của câu hỏi sử dụng LLM
        Trả về loại ý định và thông tin chi tiết
        """
        provider = provider or self.default_provider
        
        # Tạo prompt phân tích ý định
        prompt = self._create_intent_prompt(query)
        
        try:
            # Gọi AI API tương ứng
            if provider == "gemini":
                response = self.gemini_service.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                # Phân tích kết quả JSON
                return self._parse_json_response(response["answer"])
                
            elif provider == "openai":
                response = self.openai_service.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                # Phân tích kết quả JSON
                return self._parse_json_response(response["answer"])
                
            else:
                logger.warning(f"Provider không hỗ trợ: {provider}, sử dụng gemini")
                response = self.gemini_service.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                return self._parse_json_response(response["answer"])
                
        except Exception as e:
            logger.error(f"Lỗi khi phân tích ý định: {str(e)}")
            return {"intent_type": "UNKNOWN", "confidence_score": 0}
    
    def answer(self, question: str, question_type: str = None, entities: List[str] = None, context: str = None) -> Dict:
        """
        Trả lời câu hỏi dựa trên phân tích ý định:
        1. Phân tích ý định để xác định loại câu hỏi
        2. Nếu là QUESTION_ANSWERING, ưu tiên sử dụng VI-MRC
        3. Nếu VI-MRC không trả lời được, chuyển sang LLM
        """
        # Phân tích ý định nếu chưa có question_type
        if not question_type:
            intent = self.analyze_query(question)
            question_type = intent.get("intent_type", "")
            
            if not entities and "parameters" in intent and "entities" in intent["parameters"]:
                entities = intent["parameters"]["entities"]
        
        # Nếu không phải là QUESTION_ANSWERING thì chuyển sang LLM
        if question_type != "QUESTION_ANSWERING":
            return self._answer_with_llm(question, context)
            
        # Nếu không có context, tìm tài liệu liên quan
        if not context:
            search_terms = entities or self.document_store.extract_keywords(question)
            if search_terms:
                relevant_docs = self.document_store.search(" ".join(search_terms), top_k=2)
                if relevant_docs:
                    context = relevant_docs[0].content
        
        # Thử sử dụng VI-MRC nếu có context
        if context:
            try:
                vimrc_response = self.vimrc_service.answer_question(question, context)
                if vimrc_response["success"] and vimrc_response["answer"].strip():
                    return {
                        "answer": vimrc_response["answer"],
                        "source": "vimrc",
                        "confidence": vimrc_response.get("confidence", 0.8)
                    }
            except Exception as e:
                logger.error(f"Lỗi khi sử dụng VI-MRC: {str(e)}")
        
        # Fallback về LLM khi VI-MRC không trả lời được
        return self._answer_with_llm(question, context)
    
    def _answer_with_llm(self, question: str, context: str = None) -> Dict:
        """
        Trả lời câu hỏi bằng LLM (Gemini hoặc OpenAI)
        """
        prompt = question
        if context:
            prompt = f"""
            Dựa trên thông tin sau:
            
            {context}
            
            Hãy trả lời câu hỏi: {question}
            """
        
        try:
            if self.default_provider == "gemini":
                response = self.gemini_service.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    is_answer=True
                )
                return {
                    "answer": response["answer"],
                    "source": "gemini",
                    "confidence": 0.7
                }
            else:
                response = self.openai_service.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    is_answer=True
                )
                return {
                    "answer": response["answer"],
                    "source": "openai",
                    "confidence": 0.7
                }
        except Exception as e:
            logger.error(f"Lỗi khi sử dụng LLM: {str(e)}")
            fallback_provider = "openai" if self.default_provider == "gemini" else "gemini"
            
            try:
                if fallback_provider == "gemini":
                    response = self.gemini_service.chat(
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        is_answer=True
                    )
                else:
                    response = self.openai_service.chat(
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        is_answer=True
                    )
                    
                return {
                    "answer": response["answer"],
                    "source": fallback_provider,
                    "confidence": 0.7
                }
            except:
                return {
                    "answer": "Không thể trả lời câu hỏi do lỗi hệ thống.",
                    "source": "error",
                    "confidence": 0
                }
    
    def _create_intent_prompt(self, query: str) -> str:
        """
        Tạo prompt để phân tích ý định của truy vấn
        """
        return f"""
        SYSTEM:
        Bạn là hệ thống phân tích truy vấn tiếng Việt. Nhiệm vụ của bạn là xác định loại câu truy vấn.
        
        Phân loại truy vấn thành một trong hai loại sau:
        1. INFORMATION_RETRIEVAL (Truy Xuất Thông Tin): Người dùng muốn tìm kiếm dữ liệu cụ thể.
        2. QUESTION_ANSWERING (Hỏi Đáp): Người dùng đang đặt câu hỏi cần giải thích hoặc phân tích.
        
        Dựa trên phân tích của bạn, hãy trả về kết quả dưới dạng JSON với định dạng sau:
        {{
          "intent_type": "INFORMATION_RETRIEVAL hoặc QUESTION_ANSWERING",
          "confidence_score": số từ 0.0 đến 1.0,
          "parameters": {{
            // Cho INFORMATION_RETRIEVAL
            "search_keywords": ["từ khóa1", "từ khóa2"],
            "filters": ["điều kiện1", "điều kiện2"],
            // Cho QUESTION_ANSWERING
            "question_type": "loại câu hỏi (what/why/how...)",
            "entities": ["thực thể1", "thực thể2"],
            "context": "ngữ cảnh nếu có"
          }}
        }}
        
        Chỉ trả về JSON thuần túy, không có markdown hoặc code block.
        
        USER:
        {query}
        """
    
    def _parse_json_response(self, response_text: str) -> Dict:
        """
        Phân tích kết quả JSON từ phản hồi của AI
        """
        try:
            # Làm sạch kết quả nếu có code block
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                if json_end == -1:
                    cleaned_text = response_text[json_start:].strip()
                else:
                    cleaned_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                if json_end == -1:
                    cleaned_text = response_text[json_start:].strip()
                else:
                    cleaned_text = response_text[json_start:json_end].strip()
            else:
                cleaned_text = response_text.strip()
            
            # Tìm { đầu tiên và } cuối cùng
            start_index = cleaned_text.find('{')
            end_index = cleaned_text.rfind('}') + 1
            
            if start_index != -1 and end_index > start_index:
                json_text = cleaned_text[start_index:end_index]
                result = json.loads(json_text)
                return result
            else:
                return {"intent_type": "UNKNOWN", "confidence_score": 0}
                
        except Exception as e:
            logger.error(f"Lỗi phân tích JSON: {str(e)}")
            return {"intent_type": "UNKNOWN", "confidence_score": 0}


# Lớp mô phỏng service AI
class GeminiService:
    def __init__(self):
        self.model_name = "gemini-pro"
    
    def chat(self, messages, temperature=0.7, max_tokens=None, is_answer=False):
        # Mô phỏng trả về kết quả
        prompt = messages[0]["content"]
        
        # Lấy phần câu hỏi người dùng từ prompt
        user_query = ""
        if "USER:" in prompt:
            user_query = prompt.split("USER:")[-1].strip()
        else:
            # Nếu không tìm thấy, lấy 50 ký tự cuối cùng làm câu hỏi
            user_query = prompt[-50:].strip()
        
        # Nếu là yêu cầu trả lời (không phải phân tích ý định)
        if is_answer:
            if "ai" in user_query.lower() or "trí tuệ nhân tạo" in user_query.lower():
                return {
                    "answer": "AI (Artificial Intelligence) hay trí tuệ nhân tạo là lĩnh vực nghiên cứu và phát triển các hệ thống máy tính có khả năng thực hiện các nhiệm vụ đòi hỏi trí thông minh của con người như học tập, lập luận, nhận thức, giải quyết vấn đề và hiểu ngôn ngữ tự nhiên."
                }
            elif "macbook" in user_query.lower() or "apple" in user_query.lower():
                return {
                    "answer": "MacBook là dòng máy tính xách tay của Apple, được đánh giá cao về thiết kế, hiệu suất và thời lượng pin. MacBook sử dụng chip Apple Silicon (M1/M2/M3) dựa trên kiến trúc ARM tiết kiệm điện, kết hợp với hệ điều hành macOS được tối ưu đặc biệt cho phần cứng Apple."
                }
            elif "samsung" in user_query.lower():
                return {
                    "answer": "Samsung là tập đoàn đa quốc gia của Hàn Quốc, nổi tiếng với các sản phẩm điện tử tiêu dùng như điện thoại thông minh Galaxy, TV, thiết bị gia dụng và linh kiện điện tử. Samsung hiện là một trong những nhà sản xuất điện thoại lớn nhất thế giới."
                }
            else:
                return {
                    "answer": "Tôi không có thông tin cụ thể về câu hỏi này. Bạn có thể cung cấp thêm chi tiết không?"
                }
        
        # Phân tích ý định dựa trên nội dung câu hỏi
        # Tìm kiếm
        if "tìm" in user_query.lower() or "kiếm" in user_query.lower() or "ở đâu" in user_query.lower() or "có những" in user_query.lower():
            # Trích xuất từ khóa
            keywords = []
            if "điện thoại" in user_query.lower():
                keywords.append("điện thoại")
            if "laptop" in user_query.lower(): 
                keywords.append("laptop")
            if "macbook" in user_query.lower():
                keywords.append("macbook")
            if "samsung" in user_query.lower():
                keywords.append("samsung")
            if "iphone" in user_query.lower() or "apple" in user_query.lower():
                keywords.append("iphone")
            
            # Nếu không tìm thấy từ khóa cụ thể, lấy các từ có ý nghĩa
            if not keywords:
                words = user_query.lower().split()
                for word in words:
                    if len(word) > 3 and word not in ["tìm", "kiếm", "những", "những", "giúp", "cho"]:
                        keywords.append(word)
            
            # Trích xuất điều kiện lọc
            filters = []
            if "dưới" in user_query.lower() and any(str(i) in user_query for i in range(10)):
                price_text = ""
                for i, word in enumerate(user_query.split()):
                    if "dưới" in word and i+1 < len(user_query.split()):
                        next_word = user_query.split()[i+1]
                        if any(c.isdigit() for c in next_word):
                            price_text = next_word
                
                if price_text:
                    # Chuẩn hóa giá trị giá
                    price_value = ''.join(filter(str.isdigit, price_text))
                    if "triệu" in user_query.lower():
                        price_filter = f"giá < {price_value}000000"
                    else:
                        price_filter = f"giá < {price_value}"
                    filters.append(price_filter)
            
            return {
                "answer": json.dumps({
                    "intent_type": "INFORMATION_RETRIEVAL",
                    "confidence_score": 0.9,
                    "parameters": {
                        "search_keywords": keywords,
                        "filters": filters,
                        "question_type": None,
                        "entities": keywords
                    }
                }, ensure_ascii=False)
            }
        # Câu hỏi
        else:
            # Xác định loại câu hỏi
            question_type = "unknown"
            if any(w in user_query.lower() for w in ["tại sao", "vì sao", "vì lý do gì"]):
                question_type = "why"
            elif any(w in user_query.lower() for w in ["thế nào", "cách nào", "như thế nào", "làm sao", "làm thế nào"]):
                question_type = "how"
            elif any(w in user_query.lower() for w in ["là gì", "là ai", "định nghĩa", "khái niệm"]):
                question_type = "what"
            elif any(w in user_query.lower() for w in ["khi nào", "thời gian", "bao giờ"]):
                question_type = "when"
            elif any(w in user_query.lower() for w in ["ở đâu", "nơi nào", "địa điểm"]):
                question_type = "where"
            
            # Trích xuất thực thể
            entities = []
            if "ai" in user_query.lower() or "trí tuệ nhân tạo" in user_query.lower():
                entities.append("AI")
                entities.append("trí tuệ nhân tạo")
            if "điện thoại" in user_query.lower():
                entities.append("điện thoại")
            if "laptop" in user_query.lower():
                entities.append("laptop")
            if "macbook" in user_query.lower() or "apple" in user_query.lower():
                entities.append("MacBook")
            if "samsung" in user_query.lower():
                entities.append("Samsung")
            if "iphone" in user_query.lower():
                entities.append("iPhone")
            if "pin" in user_query.lower() or "hiệu suất" in user_query.lower():
                entities.append("pin")
                entities.append("hiệu suất")
            
            # Nếu không có entities cụ thể, thử extract từ quan trọng
            if not entities:
                words = user_query.lower().split()
                for word in words:
                    if len(word) > 3 and word not in ["tại", "sao", "như", "thế", "nào", "khi", "là", "gì", "ai"]:
                        entities.append(word)
            
            return {
                "answer": json.dumps({
                    "intent_type": "QUESTION_ANSWERING",
                    "confidence_score": 0.92,
                    "parameters": {
                        "search_keywords": [],
                        "filters": [],
                        "question_type": question_type,
                        "entities": entities,
                        "context": entities[0] if entities else None
                    }
                }, ensure_ascii=False)
            }

class OpenAIService:
    def __init__(self):
        self.model_name = "gpt-3.5-turbo"
    
    def chat(self, messages, temperature=0.7, max_tokens=None, is_answer=False):
        # Mô phỏng trả về kết quả giống Gemini
        return GeminiService().chat(messages, temperature, max_tokens, is_answer)

class VIMRCService:
    def __init__(self):
        self.model_name = "vi-mrc-large"
    
    def answer_question(self, question, context):
        # Nếu chứa từ MacBook thì trả lời thành công
        if "macbook" in question.lower() or "apple" in question.lower():
            return {
                "success": True,
                "answer": "MacBook có hiệu suất pin tốt hơn nhờ vào chip ARM tiết kiệm điện và hệ điều hành macOS được tối ưu hóa đặc biệt cho phần cứng Apple.",
                "confidence": 0.85
            }
        # Nếu không, mô phỏng không tìm được câu trả lời
        return {
            "success": False,
            "answer": "",
            "confidence": 0
        }

class Document:
    def __init__(self, content):
        self.content = content

class DocumentStore:
    def __init__(self):
        pass
    
    def extract_keywords(self, query):
        # Mô phỏng trích xuất từ khóa
        keywords = []
        if "điện thoại" in query.lower():
            keywords.append("điện thoại")
        if "samsung" in query.lower():
            keywords.append("Samsung")
        if "iphone" in query.lower():
            keywords.append("iPhone")
        if "laptop" in query.lower():
            keywords.append("laptop")
        if "macbook" in query.lower():
            keywords.append("MacBook")
        if "dell" in query.lower():
            keywords.append("Dell")
        if "ai" in query.lower() or "trí tuệ nhân tạo" in query.lower():
            keywords.append("AI")
        
        return keywords
    
    def search(self, query, top_k=2):
        # Mô phỏng kết quả tìm kiếm
        if "macbook" in query.lower():
            return [Document("MacBook sử dụng chip Apple Silicon dựa trên kiến trúc ARM tiết kiệm điện hơn so với chip Intel. Ngoài ra, macOS được tối ưu hóa đặc biệt cho phần cứng Apple, giúp quản lý pin hiệu quả hơn.")]
        elif "samsung" in query.lower():
            return [Document("Samsung Galaxy là dòng điện thoại Android cao cấp của Samsung, với nhiều mẫu mã đa dạng từ phân khúc giá rẻ đến cao cấp.")]
        elif "iphone" in query.lower():
            return [Document("iPhone sử dụng hệ điều hành iOS và chip A-series do Apple thiết kế, mang lại hiệu suất cao và tích hợp tốt giữa phần cứng và phần mềm.")]
        elif "ai" in query.lower() or "trí tuệ nhân tạo" in query.lower():
            return [Document("AI (Artificial Intelligence) là lĩnh vực khoa học máy tính tập trung vào việc tạo ra các hệ thống có khả năng thực hiện các nhiệm vụ thường đòi hỏi trí thông minh của con người. Bao gồm học máy, xử lý ngôn ngữ tự nhiên, thị giác máy tính và nhiều lĩnh vực khác.")]
        else:
            return []


def run_interactive_demo():
    """Chạy demo tương tác với người dùng"""
    print("\n===== DEMO INTENT ROUTER =====")
    print("Phân tích ý định và trả lời câu hỏi người dùng")
    print("Gõ 'thoát' để kết thúc demo\n")
    
    # Khởi tạo router
    router = IntentRouter()
    
    while True:
        # Nhận input từ người dùng
        user_input = input("\nNhập câu hỏi hoặc truy vấn: ")
        
        # Kiểm tra nếu người dùng muốn thoát
        if user_input.lower() in ["thoát", "exit", "quit", "q"]:
            print("Kết thúc demo. Tạm biệt!")
            break
            
        if not user_input.strip():
            continue
            
        # Phân tích ý định
        intent = router.analyze_query(user_input)
        intent_type = intent.get("intent_type", "UNKNOWN")
        confidence = intent.get("confidence_score", 0)
        
        # Hiển thị kết quả phân tích
        print(f"\n--- Kết quả phân tích ---")
        print(f"Loại ý định: {intent_type}")
        print(f"Độ tin cậy: {confidence:.2f}")
        
        if "parameters" in intent:
            params = intent["parameters"]
            if intent_type == "INFORMATION_RETRIEVAL":
                keywords = params.get("search_keywords", [])
                filters = params.get("filters", [])
                print(f"Từ khóa tìm kiếm: {', '.join(keywords) if keywords else 'Không có'}")
                print(f"Bộ lọc: {', '.join(filters) if filters else 'Không có'}")
            elif intent_type == "QUESTION_ANSWERING":
                question_type = params.get("question_type", "")
                entities = params.get("entities", [])
                print(f"Loại câu hỏi: {question_type}")
                print(f"Thực thể: {', '.join(entities) if entities else 'Không có'}")
        
        # Trả lời câu hỏi
        print(f"\n--- Câu trả lời ---")
        answer_result = router.answer(user_input)
        print(f"Nội dung: {answer_result['answer']}")
        print(f"Nguồn: {answer_result['source']}")
        print(f"Độ tin cậy: {answer_result.get('confidence', 0):.2f}")
        print("-" * 40)


# Ví dụ sử dụng
if __name__ == "__main__":
    run_interactive_demo() 