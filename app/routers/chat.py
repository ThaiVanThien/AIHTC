from fastapi import APIRouter, HTTPException, Depends, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
import os
from pathlib import Path
import time
import logging

from app.services.vimrc_service import vimrc_service
from app.services.openai_service import openai_service
from app.services.gemini_service import gemini_service
from app.services.document_store import document_store
from app.core.config import settings
from app.core.ai_config import AIProvider, ai_settings, get_all_model_names
from app.schemas.ai_schemas import SmartQARequest, SmartQAResponse
from app.routers import nlp
from app.routers import vimrc
from app.routers import cloud_ai
from app.routers import chat

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={404: {"description": "Not found"}},
)

# Cấu hình templates
templates = Jinja2Templates(directory="app/templates")

# Đảm bảo thư mục templates tồn tại
os.makedirs("app/templates", exist_ok=True)

logger = logging.getLogger(__name__)

class AIProvider(str, Enum):
    VIMRC = "vimrc"
    OPENAI = "openai"
    GEMINI = "gemini"

class Message(BaseModel):
    role: str = Field(..., description="Vai trò của người gửi tin nhắn (user, system, assistant)", example="user")
    content: str = Field(..., description="Nội dung tin nhắn", example="Chào bạn, có thể giúp tôi tìm hiểu về kế toán không?")
    context: Optional[str] = Field(None, description="Ngữ cảnh cho nội dung tin nhắn (cho VI-MRC)", example="Kế toán là một hệ thống thông tin...")

class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="Danh sách các tin nhắn trong cuộc trò chuyện")
    provider: Optional[AIProvider] = Field(AIProvider.VIMRC, description="Nhà cung cấp AI để sử dụng", example="vimrc")
    model: Optional[str] = Field(None, description="Tên mô hình cụ thể", example="vi-mrc-large")
    temperature: Optional[float] = Field(0.7, description="Nhiệt độ ảnh hưởng đến tính ngẫu nhiên", example=0.7)
    max_tokens: Optional[int] = Field(500, description="Số lượng token tối đa trong phản hồi", example=500)

class ChatResponse(BaseModel):
    content: str = Field(..., description="Nội dung phản hồi từ AI")
    provider: AIProvider = Field(..., description="Nhà cung cấp AI đã sử dụng")
    model: str = Field(..., description="Mô hình đã sử dụng")

@router.get("/", response_class=HTMLResponse, summary="Giao diện Chat AI")
async def get_chat_ui(request: Request, provider: Optional[AIProvider] = Query(AIProvider.VIMRC, description="Nhà cung cấp AI mặc định")):
    """
    Hiển thị giao diện chat để tương tác với các mô hình AI
    
    Giao diện cho phép người dùng đặt câu hỏi và cung cấp ngữ cảnh
    để nhận câu trả lời từ mô hình AI được chọn (VI-MRC, OpenAI, Gemini)
    
    - **provider**: Nhà cung cấp AI mặc định (vimrc, openai, gemini)
    """
    try:
        # Lấy danh sách model từ cấu hình AI
        models = get_all_model_names()
        
        return templates.TemplateResponse("chat.html", {
            "request": request,
            "default_provider": provider,
            "models": models
        })
    except Exception as e:
        logger.error(f"Lỗi khi tải template chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi tải giao diện chat: {str(e)}")

@router.post("/smart", response_model=SmartQAResponse, summary="Trả lời câu hỏi thông minh")
async def smart_qa(request: SmartQARequest):
    """
    Trả lời câu hỏi tự động bằng cách kết hợp VI-MRC và LLM
    
    Luồng xử lý:
    1. Phân loại loại câu hỏi (factual/analytical)
    2. Tìm tài liệu liên quan
    3. Sử dụng VI-MRC nếu là câu hỏi dựa trên sự kiện và có tài liệu liên quan
    4. Sử dụng LLM (Gemini/OpenAI) nếu là câu hỏi phân tích hoặc không có tài liệu phù hợp
    
    - **question**: Câu hỏi cần trả lời
    - **provider**: Nhà cung cấp AI sử dụng khi cần LLM (openai/gemini)
    - **model**: Tên mô hình LLM cụ thể
    - **temperature**: Nhiệt độ ảnh hưởng đến tính ngẫu nhiên của LLM
    """
    start_time = time.time()
    try:
        question = request.question
        provider = request.provider
        llm_model = request.model
        temperature = request.temperature
        
        # Bước 1: Phân loại loại câu hỏi
        question_type = document_store.classify_question_type(question)
        
        # Bước 2: Tìm tài liệu liên quan
        relevant_docs = document_store.search(question, top_k=3)
        
        # Nếu không tìm thấy tài liệu bằng tìm kiếm ngữ nghĩa, thử tìm bằng từ khóa
        if not relevant_docs:
            keywords = document_store.extract_keywords(question)
            if keywords:
                relevant_docs = document_store.keyword_search(keywords, top_k=3)
        
        # Bước 3: Quyết định sử dụng VI-MRC hay LLM
        if question_type == "factual":
            # Nếu không tìm thấy tài liệu liên quan, thử tìm context từ dữ liệu huấn luyện
            context = None
            if relevant_docs:
                context = relevant_docs[0].content
            elif hasattr(vimrc_service, 'find_training_context'):
                training_context = vimrc_service.find_training_context(question)
                if training_context:
                    context = training_context
                    logger.info(f"(smart_qa) Sử dụng context từ dữ liệu huấn luyện cho câu hỏi: {question}")
            
            # Sử dụng VI-MRC với context nếu có
            if context:
                response = vimrc_service.answer_question(question, context)
                
                if response["success"] and response["answer"].strip():
                    processing_time = time.time() - start_time
                    return SmartQAResponse(
                        answer=response["answer"],
                        source="vimrc",
                        provider="vimrc",
                        model=vimrc_service.model_name,
                        confidence=response.get("confidence"),
                        has_context=True,
                        processing_time=processing_time
                    )
            # Nếu VI-MRC không trả lời được, chuyển sang LLM
        
        # Bước 4: Sử dụng LLM cho câu hỏi phân tích hoặc khi không có tài liệu phù hợp
        if provider == AIProvider.GEMINI:
            # Chuẩn bị ngữ cảnh cho LLM từ các tài liệu (nếu có)
            context = ""
            if relevant_docs:
                context = "Dựa trên thông tin: " + "\n\n".join([doc.content for doc in relevant_docs[:2]])
            
            # Đặt model nếu có chỉ định
            if llm_model:
                gemini_service.set_model(llm_model)
                
            # Xây dựng prompt với ngữ cảnh nếu có
            prompt = question
            if context:
                prompt = f"{context}\n\nCâu hỏi: {question}\n\nHãy trả lời dựa trên thông tin trên."
                
            # Gọi Gemini API
            llm_response = await gemini_service.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=500
            )
            
            processing_time = time.time() - start_time
            return SmartQAResponse(
                answer=llm_response["answer"],
                source="llm",
                provider="gemini",
                model=gemini_service.model_name,
                has_context=bool(relevant_docs),
                processing_time=processing_time
            )
        else:  # OpenAI
            # Chuẩn bị ngữ cảnh cho LLM từ các tài liệu (nếu có)
            context = ""
            if relevant_docs:
                context = "Dựa trên thông tin: " + "\n\n".join([doc.content for doc in relevant_docs[:2]])
            
            # Đặt model nếu có chỉ định
            if llm_model:
                openai_service.set_model(llm_model)
                
            # Xây dựng prompt với ngữ cảnh nếu có
            prompt = question
            if context:
                prompt = f"{context}\n\nCâu hỏi: {question}\n\nHãy trả lời dựa trên thông tin trên."
                
            # Gọi OpenAI API
            llm_response = await openai_service.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=500
            )
            
            processing_time = time.time() - start_time
            return SmartQAResponse(
                answer=llm_response["answer"],
                source="llm",
                provider="openai",
                model=openai_service.model_name,
                has_context=bool(relevant_docs),
                processing_time=processing_time
            )
            
    except Exception as e:
        processing_time = time.time() - start_time
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý câu hỏi: {str(e)}")

@router.post("/send", response_model=ChatResponse, summary="Gửi tin nhắn tới AI")
async def send_message(request: ChatRequest):
    """
    Gửi tin nhắn đến mô hình AI và nhận phản hồi
    
    - **messages**: Danh sách các tin nhắn
    - **provider**: Nhà cung cấp AI để sử dụng (vimrc, openai, gemini)
    - **model**: Tên mô hình cụ thể
    - **temperature**: Nhiệt độ ảnh hưởng đến tính ngẫu nhiên
    - **max_tokens**: Số lượng token tối đa trong phản hồi
    
    Tự động áp dụng luồng xử lý thông minh:
    1. Phân loại loại câu hỏi (factual/analytical)
    2. Tìm tài liệu liên quan
    3. Sử dụng VI-MRC nếu là câu hỏi dựa trên sự kiện và có tài liệu liên quan
    4. Sử dụng AI Provider được chọn nếu là câu hỏi phân tích hoặc không có tài liệu phù hợp
    """
    try:
        provider = request.provider
        model = request.model
        temperature = request.temperature
        max_tokens = request.max_tokens
        
        # Lấy tin nhắn người dùng mới nhất
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="Không tìm thấy tin nhắn người dùng")
        
        last_message = user_messages[-1]
        question = last_message.content
        context = last_message.context
        
        # Bước 1: Phân loại loại câu hỏi
        question_type = document_store.classify_question_type(question)
        
        # Bước 2: Tìm tài liệu liên quan nếu chưa có context
        if not context:
            relevant_docs = document_store.search(question, top_k=2)
            
            # Nếu không tìm thấy tài liệu bằng tìm kiếm ngữ nghĩa, thử tìm bằng từ khóa
            if not relevant_docs:
                keywords = document_store.extract_keywords(question)
                if keywords:
                    relevant_docs = document_store.keyword_search(keywords, top_k=2)
                    
            # Lấy context từ tài liệu tìm được
            if relevant_docs:
                context = relevant_docs[0].content
        
        # Bước 3: Quyết định sử dụng VI-MRC hay LLM
        if (provider == AIProvider.VIMRC or question_type == "factual"):
            # Nếu không có context từ tìm kiếm document store, thử tìm từ dữ liệu huấn luyện
            if not context and hasattr(vimrc_service, 'find_training_context'):
                training_context = vimrc_service.find_training_context(question)
                if training_context:
                    context = training_context
                    logger.info(f"Sử dụng context từ dữ liệu huấn luyện cho câu hỏi: {question}")
            
            # Sử dụng VI-MRC với context
            if context:  # Chỉ dùng VI-MRC khi có context
                response = vimrc_service.answer_question(question, context)
                
                if response["success"] and response["answer"].strip():
                    return ChatResponse(
                        content=response["answer"],
                        provider=AIProvider.VIMRC,
                        model=vimrc_service.model_name
                    )
                
        # Bước 4: Sử dụng LLM cho câu hỏi phân tích hoặc khi VI-MRC không có kết quả
        # Chuẩn bị ngữ cảnh cho LLM từ các tài liệu (nếu có)
        enhanced_context = ""
        if context:
            enhanced_context = f"Dựa trên thông tin: {context}\n\n"
            
        if provider == AIProvider.OPENAI:
            # Đặt model nếu có chỉ định
            if model:
                try:
                    openai_service.set_model(model)
                except Exception as e:
                    # Nếu không đặt được model, sử dụng model mặc định
                    logger.warning(f"Không thể đặt model OpenAI {model}: {str(e)}")
                
            # Xây dựng messages với enhanced context nếu có
            messages = []
            
            # Thêm system message nếu cần
            if enhanced_context:
                messages.append({"role": "system", "content": f"Hãy sử dụng những thông tin sau đây khi trả lời: {enhanced_context}"})
            
            # Thêm các tin nhắn gốc
            for msg in request.messages:
                messages.append({"role": msg.role, "content": msg.content})
                
            # Gọi API OpenAI
            response = await openai_service.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return ChatResponse(
                content=response["answer"],
                provider=AIProvider.OPENAI,
                model=openai_service.model_name
            )
        else:  # GEMINI làm mặc định
            # Đặt model theo cấu hình mặc định từ ai_settings
            default_gemini_model = ai_settings.gemini.model_name
            
            # Nếu có model được chỉ định, thử sử dụng nó
            if model:
                try:
                    gemini_service.set_model(model)
                except Exception as e:
                    # Nếu không đặt được model, sử dụng model mặc định
                    logger.warning(f"Không thể đặt model Gemini {model}: {str(e)}")
                    gemini_service.set_model(default_gemini_model)
            else:
                # Sử dụng model mặc định
                gemini_service.set_model(default_gemini_model)
                
            # Xây dựng messages với enhanced context nếu có
            messages = []
            
            # Thêm system message nếu cần
            if enhanced_context:
                messages.append({"role": "system", "content": f"Hãy sử dụng những thông tin sau đây khi trả lời: {enhanced_context}"})
            
            # Thêm các tin nhắn gốc
            for msg in request.messages:
                messages.append({"role": msg.role, "content": msg.content})
                
            # Gọi API Gemini
            try:
                response = await gemini_service.chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                return ChatResponse(
                    content=response["answer"],
                    provider=AIProvider.GEMINI,
                    model=gemini_service.model_name
                )
            except Exception as e:
                # Nếu Gemini thất bại, thử dùng OpenAI nếu có API key
                if openai_service.api_key:
                    logger.warning(f"Gemini API lỗi: {str(e)}. Thử dùng OpenAI thay thế.")
                    response = await openai_service.chat(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    return ChatResponse(
                        content=response["answer"],
                        provider=AIProvider.OPENAI,
                        model=openai_service.model_name
                    )
                else:
                    # Nếu không có OpenAI API key, thử dùng VI-MRC với câu hỏi
                    logger.warning(f"Gemini API lỗi: {str(e)}. Không có OpenAI API key. Thử dùng VI-MRC.")
                    response = vimrc_service.answer_question(question, "")
                    
                    if response["success"] and response["answer"].strip():
                        return ChatResponse(
                            content=response["answer"],
                            provider=AIProvider.VIMRC,
                            model=vimrc_service.model_name
                        )
                    else:
                        # Nếu tất cả đều thất bại, báo lỗi
                        raise HTTPException(status_code=500, detail=f"Lỗi API: {str(e)}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý tin nhắn: {str(e)}")

@router.get("/status", response_model=Dict[str, Any], summary="Trạng thái dịch vụ Chat AI")
async def get_chat_status():
    """
    Lấy trạng thái hiện tại của tất cả dịch vụ Chat AI:
    - VI-MRC: Mô hình trả lời câu hỏi tiếng Việt
    - OpenAI: API của OpenAI
    - Gemini: API của Google
    """
    return {
        "vimrc": vimrc_service.get_status(),
        "openai": openai_service.get_status(),
        "gemini": gemini_service.get_status(),
        "document_store": {
            "document_count": len(document_store.documents)
        }
    }

@router.get("/models")
async def get_models():
    # Lấy danh sách model từ cấu hình AI
    models = get_all_model_names()
    
    # Lấy model mặc định cho từng provider
    return {
        "vimrc": {
            "models": models[AIProvider.VIMRC],
            "default": ai_settings.vimrc.model_name
        },
        "openai": {
            "models": models[AIProvider.OPENAI],
            "default": ai_settings.openai.model_name
        },
        "gemini": {
            "models": models[AIProvider.GEMINI],
            "default": ai_settings.gemini.model_name
        }
    }

@router.post("/documents", response_model=Dict[str, Any], summary="Thêm tài liệu mới")
async def add_document(content: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Thêm tài liệu mới vào kho lưu trữ để sử dụng cho tìm kiếm và trả lời câu hỏi
    
    - **content**: Nội dung tài liệu
    - **metadata**: Metadata của tài liệu (tùy chọn)
    """
    try:
        doc = document_store.add_document(content, metadata)
        return {
            "success": True,
            "message": "Đã thêm tài liệu mới thành công",
            "doc_id": doc.doc_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi thêm tài liệu: {str(e)}")

@router.get("/documents", response_model=Dict[str, Any], summary="Lấy danh sách tài liệu")
async def get_documents():
    """
    Lấy danh sách tất cả tài liệu trong kho lưu trữ
    """
    try:
        docs = [
            {
                "doc_id": doc.doc_id,
                "content_preview": doc.content[:100] + "..." if len(doc.content) > 100 else doc.content,
                "metadata": doc.metadata
            }
            for doc in document_store.documents.values()
        ]
        
        return {
            "success": True,
            "count": len(docs),
            "documents": docs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy danh sách tài liệu: {str(e)}")

@router.delete("/documents/{doc_id}", response_model=Dict[str, Any], summary="Xóa tài liệu")
async def delete_document(doc_id: str):
    """
    Xóa tài liệu theo ID
    
    - **doc_id**: ID của tài liệu cần xóa
    """
    try:
        success = document_store.delete_document(doc_id)
        if success:
            return {
                "success": True,
                "message": f"Đã xóa tài liệu {doc_id}"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy tài liệu {doc_id}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa tài liệu: {str(e)}") 