from typing import Union, Optional
from pydantic import BaseModel, Field


class ItemBase(BaseModel):
    """Base schema for Item with common attributes"""
    name: str = Field(..., example="Smartphone", description="Tên của sản phẩm")
    description: Optional[str] = Field(None, example="Điện thoại thông minh", description="Mô tả sản phẩm")
    price: float = Field(..., gt=0, example=999.99, description="Giá sản phẩm")
    tax: Optional[float] = Field(None, ge=0, example=10.5, description="Thuế áp dụng")
    

class ItemCreate(ItemBase):
    """Schema for creating a new Item"""
    pass


class ItemUpdate(ItemBase):
    """Schema for updating an existing Item"""
    name: Optional[str] = Field(None, example="Smartphone Updated", description="Tên của sản phẩm")
    price: Optional[float] = Field(None, gt=0, example=899.99, description="Giá sản phẩm")


class ItemInDBBase(ItemBase):
    """Base schema for Items in DB, includes id"""
    id: int = Field(..., example=1, description="ID của sản phẩm")
    
    class Config:
        from_attributes = True


class Item(ItemInDBBase):
    """Schema for returning an Item"""
    pass
