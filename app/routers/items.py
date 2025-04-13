from typing import List, Optional, Any
from fastapi import APIRouter, HTTPException, Path, Query, Body
from app.schemas.item import Item, ItemCreate, ItemUpdate

router = APIRouter()

# Giả lập database đơn giản
fake_items_db = {
    1: {"id": 1, "name": "iPhone", "description": "Điện thoại Apple", "price": 999.0, "tax": 10.0},
    2: {"id": 2, "name": "Samsung Galaxy", "description": "Điện thoại Samsung", "price": 899.0, "tax": 9.0},
    3: {"id": 3, "name": "Xiaomi Mi", "description": "Điện thoại Xiaomi", "price": 499.0, "tax": 5.0},
}


@router.get("/", response_model=List[Item], summary="Lấy danh sách sản phẩm")
def read_items(
    skip: int = Query(0, ge=0, description="Số lượng bản ghi bỏ qua"),
    limit: int = Query(100, ge=1, le=100, description="Số lượng bản ghi tối đa trả về")
) -> List[Item]:
    """
    Lấy danh sách sản phẩm với phân trang
    
    - **skip**: Số lượng bản ghi bỏ qua (dùng cho phân trang)
    - **limit**: Số lượng bản ghi tối đa trả về (dùng cho phân trang)
    """
    return list(fake_items_db.values())[skip : skip + limit]


@router.get("/{item_id}", response_model=Item, summary="Lấy thông tin một sản phẩm")
def read_item(
    item_id: int = Path(..., ge=1, description="ID của sản phẩm cần lấy")
) -> Item:
    """
    Lấy thông tin chi tiết của một sản phẩm dựa vào ID
    
    - **item_id**: ID của sản phẩm cần lấy thông tin
    """
    if item_id not in fake_items_db:
        raise HTTPException(status_code=404, detail="Không tìm thấy sản phẩm")
    return fake_items_db[item_id]


@router.post("/", response_model=Item, status_code=201, summary="Tạo mới sản phẩm")
def create_item(
    item: ItemCreate = Body(..., description="Thông tin sản phẩm cần tạo")
) -> Item:
    """
    Tạo một sản phẩm mới
    
    - **item**: Thông tin sản phẩm cần tạo
    """
    item_id = max(fake_items_db.keys()) + 1 if fake_items_db else 1
    item_dict = item.dict()
    item_dict["id"] = item_id
    fake_items_db[item_id] = item_dict
    return item_dict


@router.put("/{item_id}", response_model=Item, summary="Cập nhật sản phẩm")
def update_item(
    item_id: int = Path(..., ge=1, description="ID của sản phẩm cần cập nhật"),
    item: ItemUpdate = Body(..., description="Thông tin cập nhật")
) -> Item:
    """
    Cập nhật thông tin của một sản phẩm
    
    - **item_id**: ID của sản phẩm cần cập nhật
    - **item**: Thông tin cần cập nhật
    """
    if item_id not in fake_items_db:
        raise HTTPException(status_code=404, detail="Không tìm thấy sản phẩm")
    
    stored_item = fake_items_db[item_id]
    update_data = item.dict(exclude_unset=True)
    
    for field, value in update_data.items():
        if value is not None:
            stored_item[field] = value
    
    fake_items_db[item_id] = stored_item
    return stored_item


@router.delete("/{item_id}", response_model=Item, summary="Xóa sản phẩm")
def delete_item(
    item_id: int = Path(..., ge=1, description="ID của sản phẩm cần xóa")
) -> Item:
    """
    Xóa một sản phẩm
    
    - **item_id**: ID của sản phẩm cần xóa
    """
    if item_id not in fake_items_db:
        raise HTTPException(status_code=404, detail="Không tìm thấy sản phẩm")
    
    item = fake_items_db[item_id]
    del fake_items_db[item_id]
    return item
